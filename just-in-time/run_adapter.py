import torch
import os
import dill
import logging
from opendelta import AdapterModel
import numpy as np
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, f1_score, auc, roc_curve
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig, AdamW, get_linear_schedule_with_warmup
from torch.optim import Adam

from util import parse_jit_args, set_seed, build_model_tokenizer_config
from process_jitfine import JITFineDataset
from models.ConcatModel import ConcatModel
from models.SingleModel import SingleModel


logger = logging.getLogger(__name__)


def train(args, train_dataset, eval_dataset, test_dataset, mymodel):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=4)

    args.max_steps = args.epochs * len(train_dataloader)
    args.save_steps = len(train_dataloader) // 5
    args.warmup_steps = 0

    # params = []
    # for n, p in mymodel.named_parameters():
    #     if "adapter" in n or "LayerNorm" in n:
    #         params.append(p)
    # optimizer = AdamW(params, lr=args.learning_rate)

    optimizer = AdamW(mymodel.parameters(), lr=args.learning_rate)

    # for n, p in mymodel.named_parameters():
    #     if "adapter" not in n or "LayerNorm" not in n:
    #         p.requires_grad = False
    # # print(filter(lambda p: p.requires_grad, mymodel.parameters()))
    # optimizer = AdamW(filter(lambda p: p.requires_grad, mymodel.parameters()), lr=args.learning_rate)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    # multi-gpu training
    if args.n_gpu > 1:
        mymodel = torch.nn.DataParallel(mymodel, device_ids=args.available_gpu)

    best_f1 = 0
    patience = 0
    mymodel.zero_grad()

    for idx in range(args.epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_loss = 0
        tr_num = 0
        for step, batch in enumerate(bar):
            input_ids, input_mask, manual_features, label = [x.to(args.device) for x in batch]
            mymodel.train()
            prob, loss = mymodel(input_ids, input_mask, manual_features, label)
            if args.n_gpu > 1:
                loss = loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # report loss
            tr_loss += loss.item()
            tr_num += 1
            if (step + 1) % args.save_steps == 0:
                logger.warning(f"epoch {idx} step {step + 1} loss {round(tr_loss / tr_num, 5)}")
                tr_loss = 0
                tr_num = 0

            # backward
            loss.backward()
            # truncate the gradient, used to prevent exploding gradient.
            torch.nn.utils.clip_grad_norm_(mymodel.parameters(), args.max_grad_norm)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            # save model after save_steps.
            if (step + 1) % args.save_steps == 0:
                results = evaluate(args, eval_dataset, mymodel)

                if idx >= args.epochs - 2:
                    checkpoint_prefix = f"epoch_{idx}_step_{step}"
                    output_dir = os.path.join(args.output_dir, f"{checkpoint_prefix}")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = mymodel.module if hasattr(mymodel, 'module') else mymodel
                    output_file = os.path.join(output_dir, "model.bin")
                    save_content = {
                        "model_state_dict": model_to_save.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        'epoch': idx,
                        'step': step
                    }
                    torch.save(save_content, output_file)

                if results["eval_f1"] > best_f1:
                    patience = 0
                    best_f1 = results["eval_f1"]
                    checkpoint_prefix = "checkpoint-best-f1"
                    output_dir = os.path.join(args.output_dir, f"{checkpoint_prefix}")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = mymodel.module if hasattr(mymodel, 'module') else mymodel
                    output_file = os.path.join(output_dir, "model.bin")
                    save_content = {
                        "model_state_dict": model_to_save.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        'epoch': idx,
                        'step': step
                    }
                    torch.save(save_content, output_file)

                else:
                    patience += 1
                    if patience > args.patience * 5:
                        logger.info('patience greater than {}, early stop!'.format(args.patience))
                        return

            if (step + 1) % args.save_steps == 0:
                test(args, test_dataset, mymodel)


def evaluate(args, eval_dataset, mymodel):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, num_workers=4)

    pred_prob = []
    true_label = []
    mymodel.eval()

    for batch in eval_dataloader:
        input_ids, input_mask, manual_features, label = [x.to(args.device) for x in batch]
        with torch.no_grad():
            prob, loss = mymodel(input_ids, input_mask, manual_features, label)
            pred_prob.append(prob.cpu().numpy())
            true_label.append(label.cpu().numpy())

    pred_prob = np.concatenate(pred_prob, 0)
    true_label = np.concatenate(true_label, 0)
    best_threshold = args.threshold

    pred_label = [0 if x < best_threshold else 1 for x in pred_prob]

    precision = precision_score(true_label, pred_label, average="binary")
    recall = recall_score(true_label, pred_label, average="binary")
    f1 = f1_score(true_label, pred_label, average="binary")

    result = {
        "eval_recall": recall,
        "eval_precision": precision,
        "eval_f1": f1,
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def test(args, test_dataset, mymodel):
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size, num_workers=4)

    # multi-gpu evaluate
    # if args.n_gpu > 1:
    #     mymodel = torch.nn.DataParallel(mymodel, device_ids=args.available_gpu)

    pred_prob = []
    true_label = []
    mymodel.eval()

    for batch in test_dataloader:
        input_ids, input_mask, manual_features, label = [x.to(args.device) for x in batch]
        with torch.no_grad():
            prob, loss = mymodel(input_ids, input_mask, manual_features, label)
            pred_prob.append(prob.cpu().numpy())
            true_label.append(label.cpu().numpy())

    pred_prob = np.concatenate(pred_prob, 0)
    true_label = np.concatenate(true_label, 0)
    best_threshold = args.threshold

    pred_label = [0 if x < best_threshold else 1 for x in pred_prob]

    precision = precision_score(true_label, pred_label, average="binary")
    recall = recall_score(true_label, pred_label, average="binary")
    f1 = f1_score(true_label, pred_label, average="binary")

    fpr, tpr, thres = roc_curve(true_label, pred_prob)
    auc_score = auc(fpr, tpr)

    result = {
        "test_recall": recall,
        "test_precision": precision,
        "test_f1": f1,
        "auc_score": auc_score
    }

    logger.info("***** Test results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #args.n_gpu = len(args.available_gpu)
    #args.device = device
    #torch.cuda.set_device(args.available_gpu[0])
    args.n_gpu = 1
    args.device = device
    torch.cuda.set_device(0)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    set_seed(args)

    model, tokenizer, config = build_model_tokenizer_config(args)

    if args.pretrained_model in ["codebert", "graphcodebert", "unixcoder"]:
        delta_model = AdapterModel(backbone_model=model,
                                   modified_modules=['attention', '[r](\d)+\.output'],
                                   bottleneck_dim=128)
        delta_model.freeze_module(exclude=["deltas", "LayerNorm"], set_state_dict=False)
    elif args.pretrained_model in ["codet5", "codet5p-770m", "codet5p", "codet5p-2b", "codet5p-16b", "codereviewer"]:
        delta_model = AdapterModel(backbone_model=model,
                                   modified_modules=['layer.0', 'layer.2', '[r]encoder\.block\.(\d)+\.layer\.[01]'],
                                   bottleneck_dim=128)
        delta_model.freeze_module(exclude=["deltas", "layer_norm"], set_state_dict=False)
    elif args.pretrained_model in ["plbart", "plbart-large"]:
        delta_model = AdapterModel(backbone_model=model,
                                   modified_modules=['self_attn_layer_norm', 'final_layer_norm'],
                                   bottleneck_dim=128)
        delta_model.freeze_module(exclude=["deltas", "self_attn_layer_norm", "final_layer_norm"], set_state_dict=False)
    # delta_model.log()

    #mymodel = SingleModel(model, config, tokenizer, args).to(device)
    mymodel = ConcatModel(model, config, tokenizer, args).to(device)
    # print(mymodel)

    # store_path = "../datasets/jitfine"
    # with open(os.path.join(store_path, "train.pkl"), 'rb') as frb1:
    #     train_dataset = dill.load(frb1)
    # with open(os.path.join(store_path, "eval.pkl"), 'rb') as frb2:
    #     eval_dataset = dill.load(frb2)
    # with open(os.path.join(store_path, "test.pkl"), 'rb') as frb3:
    #     test_dataset = dill.load(frb3)

    if args.do_train:
        train_dataset = JITFineDataset(tokenizer, args, "train")
        eval_dataset = JITFineDataset(tokenizer, args, "eval")
        test_dataset = JITFineDataset(tokenizer, args, "test")
        train(args, train_dataset, eval_dataset, test_dataset, mymodel)

    if args.do_test:
        test_dataset = JITFineDataset(tokenizer, args, "test")
        checkpoint_prefix = 'checkpoint-best-f1/model.bin'
        output_dir = os.path.join(args.output_dir, f"{checkpoint_prefix}")
        checkpoint = torch.load(output_dir)
        mymodel.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Successfully load epoch {checkpoint['epoch']}'s model checkpoint")
        test(args, test_dataset, mymodel)


if __name__ == "__main__":
    args = parse_jit_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    main(args)
