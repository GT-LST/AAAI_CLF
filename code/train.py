import argparse
import os
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import logging
from read_data import *
from model import ClassificationXLNet
from utils import ID2CLASS
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import trange
from torch.nn import CrossEntropyLoss


logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description='AAAI CLF')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=32, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--batch_size_u', default=96, type=int, metavar='N',
                    help='train batchsize')

parser.add_argument('--max_seq_length', default=64, type=int, metavar='N',
                    help='max sequence length')

parser.add_argument('--lrmain', '--learning-rate-bert', default=0.00001, type=float,
                    metavar='LR', help='initial learning rate for bert')
parser.add_argument('--lrlast', '--learning-rate-model', default=0.001, type=float,
                    metavar='LR', help='initial learning rate for models')

parser.add_argument('--gpu', default='5,6', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--output_dir', default="test_model", type=str,
                    help='path to trained model and eval and test results')
# parser.add_argument("--model_type", default=None, type=str, required=True,
#                     help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
# parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
#                     help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
parser.add_argument('--data-path', type=str, default='./processed_data/',
                    help='path to data folders')

parser.add_argument("--tsa", action='store_true',
                    help="Set this flag if tsa.")

parser.add_argument("--uda", action='store_true',
                    help="Set this flag if uda.")
parser.add_argument("--weight_decay", default=0.0, type=float,
                    help="Weight deay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
parser.add_argument('--average', type=str, default='macro',
                    help='pos_label or macro for 0/1 classes')
parser.add_argument("--warmup_steps", default=100, type=int,
                        help="Linear warmup over warmup_steps.")
parser.add_argument("--lambda_u", default=1.0, type=float,
                    help="lambda_u for consistent loss.")
parser.add_argument("--T", default=1.0, type=float,
                    help="T for sharpening.")
parser.add_argument("--no_class", default=0, type=int,
                    help="number of class.")
parser.add_argument('--tsa_type', type=str, default='exp',
                    help='tsa type')
args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
args.n_gpu = n_gpu
logger.info("Training/evaluation parameters %s", args)

best_f1 = 0


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def print_score(output_scores, no_class):
    # logger.info("============================")
    logger.info("class {}".format(ID2CLASS[no_class]))
    result = output_scores
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))

def cycle(iterable):
    while True:
        for i in iterable:
            yield i

def main():
    global best_f1
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    # config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    model_name = "xlnet-base-cased"
    if "uncased" in model_name:
        do_lower_case = True
    else:
        do_lower_case = False

    # tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=do_lower_case)
    tokenizer = XLNetTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)

    no_class = args.no_class

    train_labeled_set, train_unlabeled_set, train_unlabeled_aug_set, val_set, test_set, n_labels = \
        get_data(args.data_path, args.max_seq_length, tokenizer, no_class)

    labeled_trainloader = Data.DataLoader(
        dataset=train_labeled_set, batch_size=args.batch_size, shuffle=True)

    unlabeled_trainloader = Data.DataLoader(
       dataset=train_unlabeled_set, batch_size=args.batch_size_u, shuffle=False)

    unlabeled_aug_trainloader = Data.DataLoader(
        dataset=train_unlabeled_set, batch_size=args.batch_size_u, shuffle=False)

    unlabeled_trainloader = cycle(unlabeled_trainloader)

    unlabeled_aug_trainloader = cycle(unlabeled_aug_trainloader)

    val_loader = Data.DataLoader(
        dataset=val_set, batch_size=512, shuffle=False)

    test_loader = Data.DataLoader(
        dataset=test_set, batch_size=512, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    with_UDA = args.uda
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("Device: %s, n_gpu: %s", device, args.n_gpu)

    n_labels = 2
    # config = config_class.from_pretrained(args.model_name_or_path, num_labels=n_labels)
    # model = model_class.from_pretrained(args.model_name_or_path, config=config)
    model = ClassificationXLNet(model_name, n_labels).cuda()
    model.to(device)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    optimizer = AdamW(
    [
        {"params": model.module.transformer.parameters(), "lr": args.lrmain},
        {"params": model.module.linear.parameters(), "lr": args.lrlast},
    ])

    train_criterion = SemiLoss(tsa_type = args.tsa_type)

    all_test_f1 = []
    test_f1 = None
    best_f1 = 0

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(labeled_trainloader))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Model = %s" % str(model_name))
    logger.info("  Do lower case = %s" % str(args.do_lower_case))
    logger.info("  UDA = %s" % str(with_UDA))
    logger.info("  LAM = %s" % str(args.lambda_u))
    logger.info("  Batch size = %d" % args.batch_size)
    logger.info("  Max seq length = %d" % args.max_seq_length)

    for epoch in trange(args.epochs, ncols=50, desc="Epoch:"):
        train(labeled_trainloader, unlabeled_trainloader, unlabeled_aug_trainloader,
              model, optimizer, train_criterion, epoch, with_UDA)

        train_output_scores, train_f1 = validate(labeled_trainloader,
                                                 model, n_labels)

        logger.info("******Epoch {}, train score******".format(epoch))
        print_score(train_output_scores, no_class)

        val_output_scores, val_f1 = validate(val_loader,
                                             model, n_labels)

        logger.info("******Epoch {}, validation score******".format(epoch))
        print_score(val_output_scores, no_class)

        if val_f1 >= best_f1:
            best_f1 = val_f1
            test_output_scores, test_f1 = validate(test_loader, model, n_labels)
            all_test_f1.append(test_f1)

            # writing results
            logger.info("******Epoch {}, test score******".format(epoch))
            output_eval_file = os.path.join(args.output_dir, "results.txt")
            with open(output_eval_file, "a") as writer:
                # logger.info("***** Eval results {} *****")
                writer.write("model = %s\n" % str(model_name))
                writer.write(
                    "total batch size=%d\n" % args.batch_size)
                writer.write("train num epochs = %d\n" % args.epochs)
                writer.write("max seq length = %d\n" % args.max_seq_length)
                writer.write("  Model = %s\n" % str(model_name))
                writer.write("  UDA = %s\n" % str(with_UDA))
                writer.write("  LAM = %s\n" % str(args.lambda_u))
                logger.info("class {}".format(ID2CLASS[no_class]))
                result = test_output_scores
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

            logger.info("Saving best model checkpoint to %s", args.output_dir)
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            torch.save(model_to_save.state_dict(), os.path.join(args.output_dir,
                                                                'best_model_{}.bin'.format(ID2CLASS[no_class])))
            # Good practice: save your training arguments together with the trained model
            torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        logger.info('Best dev f1:{}; Test f1: {}'.format(best_f1, test_f1))

    logger.info('Best dev f1:{}; Test f1: {}'.format(best_f1, test_f1))


def train(labeled_trainloader, unlabeled_trainloader, unlabeled_aug_trainloader,
          model, optimizer, criterion, epoch, with_UDA):
    model.train()

    for batch_idx, (inputs , targets) in enumerate(labeled_trainloader):
        inputs, targets = inputs.cuda(),targets.cuda(non_blocking=True)
        outputs = model(inputs)

        if with_UDA:
            (_, (unsup_x)), (unsup_aug_x,  unsup_aug_x2) = next(unlabeled_trainloader), next(unlabeled_aug_trainloader)
            unsup_x = unsup_x.cuda(non_blocking=True)
            unsup_aug_x = unsup_aug_x.cuda(non_blocking=True)
            unsup_aug_x2 = unsup_aug_x2.cuda(non_blocking=True)

            with torch.no_grad():
                orig_y_pred = model(unsup_x)
                orig_y_probas = torch.softmax(orig_y_pred, dim=-1)
                aug_y_pred = model(unsup_aug_x)
                aug_y_pred2 = model(unsup_aug_x2)
                # print("org: ", orig_y_probas[0])
                p = (torch.softmax(aug_y_pred, dim=1) + torch.softmax(aug_y_pred2, dim=1) + orig_y_probas) / 3
                # print("aug1: ", torch.softmax(aug_y_pred,dim=1)[0])
                # print("aug2: ", torch.softmax(aug_y_pred2,dim=1)[0])

                pt = p ** (1 / args.T)
                targets_u = pt / pt.sum(dim=1, keepdim=True)

            if args.T != 1:
                aug_y_pred = model(torch.cat([unsup_x, unsup_aug_x, unsup_aug_x2], dim = 0))
                # print(aug_y_pred[0])

                targets_u = torch.cat([targets_u, targets_u, targets_u], dim=0)

            loss = criterion(outputs, targets, aug_y_pred, targets_u, aug_y_pred, epoch+batch_idx/len(labeled_trainloader))
            total_loss = loss[0]
        else:
            loss = criterion(outputs, targets, epoch)
            total_loss = loss

        optimizer.zero_grad()
        if batch_idx % 50 == 1:
            if with_UDA:
                print('\nepoch {}, step {}, loss_total {}, uda_loss {}, mse_loss {}'.format(
                    epoch, batch_idx, loss[0], loss[2], loss[1]))
            else:
                print('\nepoch {}, step {}, loss_total {}'.format(
                    epoch, batch_idx, loss))
        total_loss.backward()
        optimizer.step()
        # break
        # scheduler.step()

def validate(val_loader, model):
    model.eval()

    predict_dict = [0,0]
    correct_dict = [0,0]
    correct_total = [0,0]

    outputs = None
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            logits = model(inputs) # (bsz, 6, 2)
            if outputs is None:
                outputs = logits.detach().cpu().numpy()
                out_label_ids = targets.detach().cpu().numpy()
            else:
                outputs = np.append(outputs, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, targets.detach().cpu().numpy(), axis=0)

    pred = np.argmax(outputs, axis=-1)
    batch_size = pred.shape[0]
    for b in range(batch_size):
        predict_dict[int(pred[b])] += 1
        correct_dict[int(out_label_ids[b])] += 1
        if pred[b] == out_label_ids[b]:
            correct_total[int(pred[b])] += 1
    acc = simple_accuracy(pred, out_label_ids)
    # print("c_dict: ", correct_dict)
    n_class = 2
    averaging = args.average
    precision = []
    recall = []
    f1 = []

    for j in range(n_class):
        if predict_dict[j] == 0:
            p = 0
        else:
            p = correct_total[j] / predict_dict[j]
        if correct_dict[j] == 0:
            r = 0
        else:
            r = correct_total[j] / correct_dict[j]
        if p+r == 0:
            f = 0
        else:
            f = 2 * p * r / (p + r)

        precision.append(p)
        recall.append(r)
        f1.append(f)

    if averaging == "pos_label":
        p, r, f = precision[1], recall[1], f1[1]
    elif averaging == "macro":
        p, r, f = sum(precision)/n_class, \
                  sum(recall)/n_class, sum(f1)/n_class
    else:
        raise ValueError("UnsupportedOperationException")

    output_scores = {"precision":p, "recall":r, "f1":f, "acc":acc}

    output_f1 = f

    return output_scores, output_f1


def TSA(epoch, n_class, tsa_type = 'exp'):
    epoch = math.floor(epoch)/args.epochs
    if tsa_type == 'exp':
        return np.exp((epoch - 1) * 5) * (1-1/n_class) + 1/n_class
    elif tsa_type == 'linear':
        return epoch * (1- 1/n_class) + 1/n_class
    elif tsa_type == 'log':
        return (1-np.exp(-epoch * 5)) * (1-1/n_class) + 1/n_class
    else:
        return 1


def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


class SemiLoss(object):
    def __init__(self, n_labels=2, tsa_type = 'exp', weight=[1.0,1.0]):
        self.n_labels = n_labels
        self.weight = torch.tensor(weight).cuda()
        self.loss_fct = CrossEntropyLoss(weight=self.weight)
        self.tsa_type = tsa_type

    def __call__(self, outputs_x, targets_x, outputs_u=None, targets_u=None, outputs_u_2=None, epoch=None):
        loss = self.loss_fct(outputs_x.view(-1, self.n_labels), targets_x.view(-1))

        if args.tsa:
            thres = TSA(epoch, self.n_labels, self.tsa_type)
            q_y_softmax = F.softmax(outputs_x, dim=1)
            q_y_log_softmax = F.log_softmax(outputs_x, dim=1)
            count = 0
            classification_loss = 0
            for i in range(outputs_x.shape[0]):
                if q_y_softmax[i][targets_x[i].long()] < thres:
                    classification_loss += (-1 * q_y_log_softmax[i][targets_x[i].long()])
                    count += 1
            if count > 0:
                classification_loss = classification_loss / count
            else:
                classification_loss = 0
            loss = classification_loss

        if args.uda:
            probs_u = torch.softmax(outputs_u, dim=1)
            Lu = F.kl_div(probs_u.log(), targets_u, None, None, 'batchmean')
            total_loss = loss + args.lambda_u * linear_rampup(epoch) * Lu

            final_loss = (total_loss, loss, args.lambda_u * linear_rampup(epoch) * Lu)
        else:
            final_loss = loss


        return final_loss


if __name__ == "__main__":
    main()
