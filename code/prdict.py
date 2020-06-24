import argparse
import os
import torch.utils.data as Data
from transformers import *
import logging
from read_data import *
from model import ClassificationXLNet
from utils import ID2CLASS



logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description='AAAI CLF')

parser.add_argument('--batch_size', default=512, type=int, metavar='N',
                    help='eval batchsize')

parser.add_argument('--max_seq_length', default=64, type=int, metavar='N',
                    help='max sequence length')

parser.add_argument('--gpu', default='5,6', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--output_dir', default="test_model", type=str,
                    help='path to trained model and eval and test results')

parser.add_argument('--data_path', type=str, default='./processed_data/',
                    help='path to data folders')

parser.add_argument('--average', type=str, default='macro',
                    help='pos_label or macro for 0/1 classes')


parser.add_argument("--no_class", default=0, type=int,
                    help="number of class.")

parser.add_argument('--model_path', type=str, default='exp',
                    help='path')
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
    if "uncased" in model_name:
        do_lower_case = True
    else:
        do_lower_case = False
    tokenizer = XLNetTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)
    no_class = args.no_class

    train_labeled_set, train_unlabeled_set, train_unlabeled_aug_set, val_set, test_set, n_labels = \
        get_data(args.data_path, args.max_seq_length, tokenizer, no_class, True)


    test_loader = Data.DataLoader(
        dataset=test_set, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("Device: %s, n_gpu: %s", device, args.n_gpu)

    n_labels = 2
    # config = config_class.from_pretrained(args.model_name_or_path, num_labels=n_labels)
    # model = model_class.from_pretrained(args.model_name_or_path, config=config)
    model = ClassificationXLNet(model_name, n_labels).cuda()
    model_path = os.path.join(args.model_path, 'best_model_{}.bin'.format(ID2CLASS[no_class]))
    model_state_dict = torch.load(model_path)
    model.load_state_dict(model_state_dict)

    model.to(device)
    validate(test_loader, model, mode = 'Test')


def validate(val_loader, model, mode):
    model.eval()

    predict_dict = [0,0]
    correct_dict = [0,0]
    correct_total = [0,0]
    outputs = None
    if mode == 'Test':
        with torch.no_grad():
            for batch_idx, (_, inputs) in enumerate(val_loader):
                inputs = inputs.cuda()
                logits = model(inputs)  # (bsz, 6, 2)
                if outputs is None:
                    outputs = logits.detach().cpu().numpy()
                else:
                    outputs = np.append(outputs, logits.detach().cpu().numpy(), axis=0)
        pred = np.argmax(outputs, axis=-1)
        index = 0
        print("Start writing results.")
        with open(os.path.join(args.output_dir, "predict.csv"), "w") as writer:
            writer.write("index" + "," + "prediction" + "\n")
            for p in pred:
                writer.write(str(index) + "," + str(p) + "\n")
                index += 1
        return
    else:
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
    # logger.info("averaging method: {}".format(averaging))

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


if __name__ == "__main__":
    main()
