import torch
import os
import dill
import logging
import multiprocessing
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from sklearn.metrics import recall_score, precision_score, f1_score, auc, roc_curve, confusion_matrix
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler, SubsetRandomSampler
#from transformers import RobertaModel, RobertaTokenizer, RobertaConfig, AdamW, get_linear_schedule_with_warmup
from transformers import get_linear_schedule_with_warmup
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim.adamw import AdamW

from util import parse_jit_args, set_seed, build_model_tokenizer_config
from process_jitfine import JITFineDataset, JITFineMessageManualDataset, JITFineDatasetWithTextManualFeatures
from models.SingleModel import SingleModel
from models.ConcatModel import ConcatModel
from models.ManualModel import ManualModel
from process_jitfine import JITFineManualDataset

import pandas as pd

from skewed_oversample import SkewedRandomSampler, update_orb
import pickle
import matplotlib.pyplot as plt
from river import metrics

logger = logging.getLogger(__name__)


def train(args, train_dataset, eval_dataset, mymodel):
    if args.oversample: 
        targets = train_dataset.get_targets()
        class_counts = np.bincount(targets)
        assert len(class_counts) == 2
        logger.info(f"class_counts: {class_counts}")
        class_weights = 1. / class_counts
        logger.info(f"class_weights: {class_weights}")
        sample_weights = class_weights[targets]
        train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=int(class_counts[0] + class_counts[1]), replacement=True)
    elif args.skewed_oversample: 
        train_sampler = SkewedRandomSampler(train_dataset, len(train_dataset))
    elif args.undersample: 
        targets = train_dataset.get_targets()
        class_counts = np.bincount(targets)
        assert len(class_counts) == 2
        logger.info(f"class_counts: {class_counts}")
        min_class_count = np.min(class_counts)
        indices_to_keep = []
        
        for class_idx in range(len(class_counts)):
            class_indices = [i for i, x in enumerate(targets) if x == class_idx]
            class_indices = np.array(class_indices)
            # Randomly choose 'min_class_count' number of indices from this class
            sampled_class_indices = np.random.choice(class_indices, min_class_count, replace=False)
            indices_to_keep.extend(sampled_class_indices)
        
        print(f"Number of indices to keep: {len(indices_to_keep)}")
        train_sampler = SubsetRandomSampler(indices_to_keep)
    else:
        train_sampler = RandomSampler(train_dataset)
    
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=4)

    args.max_steps = args.epochs * len(train_dataloader)

    args.save_steps = max(len(train_dataloader), len(train_dataloader) // 5)
    args.warmup_steps = 0

    optimizer = AdamW(mymodel.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    # multi-gpu training
    if args.n_gpu > 1:
        mymodel = torch.nn.DataParallel(mymodel, device_ids=args.available_gpu)

    #best_f1 = 0
    best = 0

    # Evaluate before training
    if args.online_mode and os.path.exists(f"{args.output_dir}/checkpoint-best-{args.eval_metric}/model.bin"):
        logger.info("[ONLINE MODE] Evaluating before training...")
        if args.eval_metric == "f1":
            results = evaluate(args, eval_dataset, mymodel)
            best = results["eval_f1"]
        else:
            results = evaluate_gmean(args, eval_dataset, mymodel)
            best = results["g_mean"]

    patience = 0
    mymodel.zero_grad()

    model_changed = False

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
                if args.eval_metric == "f1":
                    results = evaluate(args, eval_dataset, mymodel)
                else:
                    results = evaluate_gmean(args, eval_dataset, mymodel)

                if args.eval_metric == "f1" and results["eval_f1"] > best:
                    patience = 0
                    best = results["eval_f1"]
                    checkpoint_prefix = "checkpoint-best-f1"
                    output_dir = os.path.join(args.output_dir, f"{checkpoint_prefix}")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = mymodel.module if hasattr(mymodel, 'module') else mymodel
                    output_file = os.path.join(output_dir, "model.bin")
                    save_content = {
                        "model_state_dict": model_to_save.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict()
                    }
                    torch.save(save_content, output_file)
                    model_changed = True
                elif args.eval_metric == "gmean" and "g_mean" in results and results["g_mean"] > best:
                    patience = 0
                    best = results["g_mean"]
                    checkpoint_prefix = "checkpoint-best-gmean"
                    output_dir = os.path.join(args.output_dir, f"{checkpoint_prefix}")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = mymodel.module if hasattr(mymodel, 'module') else mymodel
                    output_file = os.path.join(output_dir, "model.bin")
                    save_content = {
                        "model_state_dict": model_to_save.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict()
                    }
                    logger.info("Saving model to {}".format(output_file))
                    torch.save(save_content, output_file)
                    model_changed = True
                elif args.eval_metric == "gmean" and "eval_loss" in results and (results["eval_loss"] < best or best == 0):
                    patience = 0
                    best = results["eval_loss"]
                    checkpoint_prefix = "checkpoint-best-gmean"
                    output_dir = os.path.join(args.output_dir, f"{checkpoint_prefix}")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = mymodel.module if hasattr(mymodel, 'module') else mymodel
                    output_file = os.path.join(output_dir, "model.bin")
                    save_content = {
                        "model_state_dict": model_to_save.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict()
                    }
                    torch.save(save_content, output_file)
                    model_changed = True
                else:
                    patience += 1
                    if patience > args.patience * 5:
                        logger.info('patience greater than {}, early stop!'.format(args.patience))
                        return
                    
        #skewed oversampler 
        if args.skewed_oversample:
            update_sampler(args, train_dataset, mymodel, train_sampler)
          
    if model_changed:
        update_training_status(args, "changed")
    else:
        update_training_status(args, "unchanged")

def update_sampler(args, train_dataset, mymodel, train_sampler):
    logger.info(f"Updating sampler. Window size = {args.window_size}, target_th = {args.target_th}, l0 = {args.l0}, l1 = {args.l1}, m = {args.m}")
    results_ma = predict_ma(args, train_dataset[-args.window_size:], mymodel)
    orb0, orb1 = update_orb(results_ma, args.target_th, args.l0, args.l1, args.m)
    train_sampler.obf0 = orb0
    train_sampler.obf1 = orb1

def update_training_status(args, status):
    with open(os.path.join(args.output_dir, "training_status.txt"), "w") as log_file:
        log_file.write(status)

def predict_ma(args, eval_dataset, mymodel):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, num_workers=4)

    pred_prob = []
    mymodel.eval()

    for batch in eval_dataloader:
        input_ids, input_mask, manual_features, label = [x.to(args.device) for x in batch]
        with torch.no_grad():
            prob, loss = mymodel(input_ids, input_mask, manual_features, label)
            pred_prob.append(prob.cpu().numpy())

    pred_prob = np.concatenate(pred_prob, 0)

    pred_label = [0 if x < 0.5 else 1 for x in pred_prob]

    return pred_label

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

    print(pred_prob)
    pred_label = [0 if x < best_threshold else 1 for x in pred_prob]

    precision = precision_score(true_label, pred_label, average="binary")
    recall = recall_score(true_label, pred_label, average="binary")
    f1 = f1_score(true_label, pred_label, average="binary")

    fpr, tpr, thres = roc_curve(true_label, pred_prob)
    auc_score = auc(fpr, tpr)

    result = {
        "eval_recall": recall,
        "eval_precision": precision,
        "eval_f1": f1,
        "auc_score": auc_score
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result

def evaluate_gmean(args, eval_dataset, mymodel):
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

    if true_label.sum() > 0 and true_label.sum() < len(true_label):
        best_threshold = args.threshold

        pred_label = [0 if x < best_threshold else 1 for x in pred_prob]

        precision = precision_score(true_label, pred_label, average="binary")
        recall = recall_score(true_label, pred_label, average="binary")
        f1 = f1_score(true_label, pred_label, average="binary")

        fpr, tpr, thres = roc_curve(true_label, pred_prob)
        auc_score = auc(fpr, tpr)

        # Calculate G-Mean
        tn, fp, fn, tp = confusion_matrix(true_label, pred_label).ravel()
        g_mean = np.sqrt((tp / (tp + fn)) * (tn / (tn + fp)))
        r1 = tp / (tp + fn)
        assert r1 == recall

        result = {
            "eval_recall": recall,
            "eval_precision": precision,
            "eval_f1": f1,
            "auc_score": auc_score,
            "g_mean": g_mean,
        }

    else:
        logger.info("No positive examples -> using validation loss...")
        result = {"eval_loss": loss}
    
    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(float(result[key]), 4)))

    return result

def find_best_threshold(fpr, tpr, thresholds):
    distances = np.sqrt((fpr - 0)**2 + (tpr - 1)**2)
    best_index = np.argmin(distances)
    return {
        'best_threshold': thresholds[best_index],
        'fpr': fpr[best_index],
        'tpr': tpr[best_index],
        'distance': distances[best_index],
        'fpr_all': fpr,
        'tpr_all': tpr
    }

def test(args, test_dataset, mymodel):
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size, num_workers=4)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        mymodel = torch.nn.DataParallel(mymodel, device_ids=args.available_gpu)

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

    if args.calculate_metrics:
        assert true_label.sum() > 0

    logger.info("best_threshold = " + str(best_threshold))
    pred_label = [0 if x < best_threshold else 1 for x in pred_prob]

    if args.calculate_metrics: 
        # Save results to excel
        results_df = pd.DataFrame({'pred_prob': pred_prob.squeeze(), 'true_label': true_label, 'pred_label': pred_label})
        results_df.to_excel('results.xlsx', index=False)

        precision = precision_score(true_label, pred_label, average="binary")
        recall = recall_score(true_label, pred_label, average="binary")
        f1 = f1_score(true_label, pred_label, average="binary")
        fpr, tpr, thres = roc_curve(true_label, pred_prob)
        auc_score = auc(fpr, tpr)

        # Calculate G-Mean
        tn, fp, fn, tp = confusion_matrix(true_label, pred_label, labels=[0,1]).ravel() e

        g_mean = 0
        r1 = 0
        r0 = 0

        if tp + fn > 0 and tn + fp > 0:    
            g_mean = np.sqrt((tp / (tp + fn)) * (tn / (tn + fp)))
            r1 = tp / (tp + fn)
                
            assert r1 == recall
            r0 = tn / (tn + fp)

        #Rolling ROC AUC
        metric = metrics.RollingROCAUC()
        for yt, yp in zip(true_label, pred_prob):
            metric.update(yt, yp)

        result = {
            "test_recall": recall,
            "test_precision": precision,
            "test_f1": f1,
            "auc_score": auc_score,
            "rolling_auc": metric.get(),
            "g_mean": g_mean,
            "R0": r0,
            "R1": r1
        }

        with open('results.pkl', 'wb') as f:
            pickle.dump(result, f)

        logger.info("***** Test results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))

    predictions = {'pred_label': pred_label, 'true_label': true_label, 'pred_prob': pred_prob.tolist()}

    with open('predictions.pkl', 'wb') as f:
        pickle.dump(predictions, f)

def main_test(args):
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
    if args.pretrained_model in ["codet5", "codet5p-770m", "codet5p", "codet5p-2b", "codet5p-16b"]:
        peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=64, lora_alpha=32,
                                 lora_dropout=0.1, target_modules=["q", "v"])
    elif args.pretrained_model in ["plbart"]:
        peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=64, lora_alpha=32,
                                 lora_dropout=0.1, target_modules=["q_proj", "v_proj"])
    elif args.pretrained_model in ["codebert", "graphcodebert", "unixcoder"]:
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=64, lora_alpha=32,
                                 lora_dropout=0.1, target_modules=["query", "value"])
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

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
        #
        #train_dataset = JITFineDataset(tokenizer, args, "train")
        #eval_dataset = JITFineDataset(tokenizer, args, "eval")
        train_dataset = JITFineMessageManualDataset(tokenizer, args, "train")
        eval_dataset = JITFineMessageManualDataset(tokenizer, args, "eval")
        train(args, train_dataset, eval_dataset, mymodel)

    if args.do_test:
        #test_dataset = JITFineDataset(tokenizer, args, "test")
        test_dataset = JITFineMessageManualDataset(tokenizer, args, "test")
        checkpoint_prefix = 'checkpoint-best-f1/model.bin'
        output_dir = os.path.join(args.output_dir, f"{checkpoint_prefix}")
        checkpoint = torch.load(output_dir)
        mymodel.load_state_dict(checkpoint['model_state_dict'])
        test(args, test_dataset, mymodel)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = 1
    args.device = device
    torch.cuda.set_device(0)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    set_seed(args)

    model, tokenizer, config = build_model_tokenizer_config(args)
    if args.pretrained_model in ["codet5", "codet5p-770m", "codet5p", "codet5p-2b", "codet5p-16b", "codereviewer"]:
        peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=64, lora_alpha=32,
                                 lora_dropout=0.1, target_modules=["q", "v"])
    elif args.pretrained_model in ["plbart"]:
        peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=64, lora_alpha=32,
                                 lora_dropout=0.1, target_modules=["q_proj", "v_proj"])
    elif args.pretrained_model in ["codebert", "graphcodebert", "unixcoder"]:
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=64, lora_alpha=32,
                                 lora_dropout=0.1, target_modules=["query", "value"])
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    if args.only_manual:
        mymodel = ManualModel(args).to(device)
    else:
        mymodel = ConcatModel(model, config, tokenizer, args).to(device)
        #mymodel = SingleModel(model, config, tokenizer, args).to(device)

    if args.do_train:
        logger.info("Training for the first time...")
        if args.only_manual:
            train_dataset = JITFineManualDataset(args, "train")
            eval_dataset = JITFineManualDataset(args, "eval")
        else:
            train_dataset = JITFineDataset(tokenizer, args, "train")
            eval_dataset = JITFineDataset(tokenizer, args, "eval")
        train(args, train_dataset, eval_dataset, mymodel)
        
    if args.do_resume_training:
        logger.info("Resuming training...")
        #checkpoint_prefix = 'checkpoint-best-f1/model.bin'
        checkpoint_prefix = f'checkpoint-best-{args.eval_metric}/model.bin'
        output_dir = os.path.join(args.output_dir, f"{checkpoint_prefix}")
        checkpoint = torch.load(output_dir)
        mymodel.load_state_dict(checkpoint['model_state_dict'])
        if args.only_manual:
            train_dataset = JITFineManualDataset(args, "train")
            eval_dataset = JITFineManualDataset(args, "eval")
        else:
            train_dataset = JITFineDataset(tokenizer, args, "train")
            eval_dataset = JITFineDataset(tokenizer, args, "eval")
        train(args, train_dataset, eval_dataset, mymodel)
    
    if args.do_test:
        if args.only_manual:
            test_dataset = JITFineManualDataset(args, "test")
        else:
            test_dataset = JITFineDataset(tokenizer, args, "test")
        #checkpoint_prefix = 'checkpoint-best-f1/model.bin'
        checkpoint_prefix = f'checkpoint-best-{args.eval_metric}/model.bin'
        output_dir = os.path.join(args.output_dir, f"{checkpoint_prefix}")
        checkpoint = torch.load(output_dir)
        mymodel.load_state_dict(checkpoint['model_state_dict'])
        test(args, test_dataset, mymodel)

if __name__ == "__main__":
    args = parse_jit_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logger.setLevel(logging.INFO)
    main(args)
    # M + EF
    #main_test(args)