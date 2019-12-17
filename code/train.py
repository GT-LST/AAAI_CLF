import argparse
import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from pytorch_transformers import *
from torch.autograd import Variable
from torch.utils.data import Dataset
import logging
from read_data import *
from model import ClassificationXLNet
from utils import ALL_MODELS, ID2CLASS, MODEL_CLASSES
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss, MSELoss

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description='AAAI CLF')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
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
parser.add_argument("--do_lower_case", action='store_true',
                    help="Set this flag if you are using an uncased model.")
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
parser.add_argument("--lam", default=1.0, type=float,
                    help="lam for uda loss.")
parser.add_argument("--no_class", default=0, type=int,
                    help="number of class.")
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

    # if "uncased" in args.model_name_or_path:
    #     args.do_lower_case = True
    # else:
    #     args.do_lower_case = False

    # tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
    model_name = "xlnet-base-cased"
    tokenizer = XLNetTokenizer.from_pretrained(model_name)
    # model_name = args.model_name_or_path
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
    lam = args.lam
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("Device: %s, n_gpu: %s", device, args.n_gpu)

    n_labels = 2
    # config = config_class.from_pretrained(args.model_name_or_path, num_labels=n_labels)
    # model = model_class.from_pretrained(args.model_name_or_path, config=config)
    model = ClassificationXLNet(model_name, n_labels).cuda()
    # model = ClassificationXLNet(model, n_labels).cuda()

    model.to(device)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    optimizer = AdamW(
    [
        {"params": model.module.xlnet.parameters(), "lr": args.lrmain},
        {"params": model.module.linear.parameters(), "lr": args.lrlast},
    ])
    # Prepare optimizer and schedule (linear warmup and decay)
    # no_decay = ['bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #      'weight_decay': args.weight_decay},
    #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]
    # optimizer = AdamW(optimizer_grouped_parameters, lr=args.lrmain, eps=args.adam_epsilon)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
    #                                             num_training_steps=len(train_labeled_set))

    train_criterion = SemiLoss()
    # consistency_criterion = nn.KLDivLoss(reduction='batchmean')

    all_test_f1 = []
    test_f1 = None

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(labeled_trainloader))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Model = %s" % str(model_name))
    logger.info("  Do lower case = %s" % str(args.do_lower_case))
    logger.info("  UDA = %s" % str(with_UDA))
    logger.info("  LAM = %s" % str(lam))
    # logger.info("  Lower case = %s" % str(args.do_lower_case))
    logger.info("  Batch size = %d" % args.batch_size)
    logger.info("  Max seq length = %d" % args.max_seq_length)

    for epoch in trange(args.epochs, ncols=50, desc="Epoch:"):
        train(labeled_trainloader, unlabeled_trainloader, unlabeled_aug_trainloader,
              model, optimizer, train_criterion, consistency_criterion, epoch, with_UDA, lam)

        train_output_scores, train_f1 = validate(labeled_trainloader,
                                                 model, n_labels, mode='Train Stats')

        logger.info("******Epoch {}, train score******".format(epoch))
        print_score(train_output_scores, no_class)
        # print("epoch {}, train f1 {}".format(epoch, train_f1))

        val_output_scores, val_f1 = validate(val_loader,
                                             model, n_labels, mode='Valid Stats')

        logger.info("******Epoch {}, validation score******".format(epoch))
        print_score(val_output_scores, no_class)
        # print("epoch {}, val f1 {}".format(epoch, val_f1))

        if val_f1 >= best_f1:
            best_f1 = val_f1
            test_output_scores, test_f1 = validate(test_loader, model, n_labels, mode = 'Test')
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
                writer.write("  Model = %s" % str(model_name))
                writer.write("  UDA = %s" % str(with_UDA))
                writer.write("  LAM = %s" % str(lam))
                logger.info("class {}".format(ID2CLASS[no_class]))
                result = test_output_scores
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

            logger.info("Saving best model checkpoint to %s", args.output_dir)
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            # model_to_save = model.module if hasattr(model,
            #                                         'module') else model  # Take care of distributed/parallel training
            # model_to_save.save_pretrained(args.output_dir)
            # tokenizer.save_pretrained(args.output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            # output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
            # torch.save(model_to_save.state_dict(), output_model_file)
            torch.save(model_to_save.state_dict(), os.path.join(args.output_dir, 'best_model.bin'))
            # Good practice: save your training arguments together with the trained model
            torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        logger.info('Best dev f1:{}; Test f1: {}'.format(best_f1, test_f1))

    logger.info('Best dev f1:{}; Test f1: {}'.format(best_f1, test_f1))


def train(labeled_trainloader, unlabeled_trainloader, unlabeled_aug_trainloader,
          model, optimizer, criterion, consistency_criterion, epoch, with_UDA, lam):
    model.train()

    for batch_idx, (inputs , targets) in enumerate(labeled_trainloader):
        inputs, targets = inputs.cuda(),targets.cuda(non_blocking=True)
        # print("input: ", inputs)
        # print("tgt: ", targets)
        # outputs = model(inputs)
        # batch, targets = tuple(t.to(device) for t in inputs), targets.to(device)
        # inputs = {'input_ids': batch[0],
        #           'attention_mask': batch[1],
        #           'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
        #           XLM don't use segment_ids
        #           }
        outputs = model(inputs)
        # outputs = outputs[0]
        # print("outputs: ", outputs.shape)
        # print("targets: ", targets.shape)
        # print(len(outputs))
        # print(outputs)
        loss = criterion(outputs, targets, epoch)

        if with_UDA:
            unsup_x, unsup_aug_x = next(unlabeled_trainloader), next(unlabeled_aug_trainloader)
            unsup_x = unsup_x.cuda(non_blocking=True)
            unsup_aug_x = unsup_aug_x.cuda(non_blocking=True)
            # with torch.no_grad():
            unsup_orig_y_pred = model(unsup_x).detach()
            unsup_orig_y_probas = torch.softmax(unsup_orig_y_pred, dim=-1)

            unsup_aug_y_pred = model(unsup_aug_x)
            unsup_aug_y_probas = torch.log_softmax(unsup_aug_y_pred, dim=-1)

            consistency_loss = consistency_criterion(unsup_aug_y_probas, unsup_orig_y_probas)

        final_loss = loss

        if with_UDA:
            uda_loss = lam * consistency_loss
            final_loss += uda_loss
        # print("outputs: ", outputs[0])
        # print("targets: ", targets[0])
        # outputs = torch.sigmoid(outputs)
        # print("o: ", outputs[0])
        # id_1, id_0 = torch.where(outputs > 0.5), torch.where(outputs < 0.5)
        # outputs[id_1] = 1
        # outputs[id_0] = 0
        # outputs = outputs.detach().cpu().numpy()
        # targets = targets.to('cpu').numpy()
        # print("outputs: ", outputs[:20, 0])
        # print("targets: ", targets[:20, 0])

        optimizer.zero_grad()
        if batch_idx % 50 == 1:
            if with_UDA:
                print('\nepoch {}, step {}, loss_total {}, uda_loss {}, mse_loss {}'.format(
                    epoch, batch_idx, final_loss.item(), uda_loss.item(), loss.item()))
            else:
                print('\nepoch {}, step {}, loss_total {}'.format(
                    epoch, batch_idx, final_loss.item()))
        final_loss.backward()
        optimizer.step()
        # scheduler.step()

def validate(val_loader, model, n_labels, mode):
    model.eval()

    predict_dict = [0,0]
    correct_dict = [0,0]
    correct_total = [0,0]

    outputs = None
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            logits = model(inputs)
            # batch, targets = tuple(t.to(device) for t in inputs), targets.to(device)
            # inputs = {'input_ids': batch[0],
            #           'attention_mask': batch[1],
            #           'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                      # XLM don't use segment_ids
                      # }
            # logits = model(**inputs)[0]
            # logits = model(inputs)
            # outputs = torch.sigmoid(outputs[0])
            # outputs = torch.sigmoid(outputs)
            # outputs = np.argmax(outputs, axis=1)
            if outputs is None:
                outputs = logits.detach().cpu().numpy()
                out_label_ids = targets.detach().cpu().numpy()
            else:
                outputs = np.append(outputs, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, targets.detach().cpu().numpy(), axis=0)

    pred = np.argmax(outputs, axis=1)
    batch_size = pred.shape[0]
    for b in range(batch_size):
        predict_dict[int(pred[b])] += 1
        correct_dict[int(out_label_ids[b])] += 1
        if pred[b] == out_label_ids[b]:
            correct_total[int(pred[b])] += 1
    acc = simple_accuracy(pred, out_label_ids)

    n_class = 2
    averaging = args.average
    # logger.info("averaging method: {}".format(averaging))

    precision, recall, f1 = [], [], []
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


class SemiLoss(object):
    def __init__(self, n_labels=2):
        self.n_labels = n_labels

    def __call__(self, outputs_x, targets_x, epoch):
        loss_fct = CrossEntropyLoss()
        # print("output_x: ", outputs_x.shape)
        # print("targets_x: ", targets_x.shape)
        loss = loss_fct(outputs_x.view(-1, self.n_labels), targets_x.view(-1))

        return loss
        # Lx = - torch.mean(torch.sum(F.logsigmoid(outputs_x) * targets_x, dim=1))
        # return Lx

if __name__ == "__main__":
    main()
