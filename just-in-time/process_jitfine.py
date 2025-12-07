import os
import random
import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn import preprocessing
from transformers import RobertaTokenizer
import re

from util import parse_jit_args, build_model_tokenizer_config, set_seed


def normalize_df(df, feature_columns):
    df["fix"] = df["fix"].apply(lambda x: float(bool(x)))
    df = df.astype({i: "float32" for i in feature_columns})
    return df[["commit_hash"] + feature_columns]

def parse_data_file(args, mode):
    codechange_file = ""
    feature_file = ""
    if mode == "train":
        codechange_file, feature_file = args.train_data_file
    elif mode == "eval":
        codechange_file, feature_file = args.eval_data_file
    elif mode == "test":
        codechange_file, feature_file = args.test_data_file

    ccdata = pd.read_pickle(codechange_file)
    fedata = pd.read_pickle(feature_file)
    
    examples = process_fedata_ccdata(fedata, ccdata)

    if mode == "train":
        random.seed(args.seed)
        random.shuffle(examples)

    return examples

def parse_ma_data(args):
    codechange_file = ""
    feature_file = ""
    codechange_file, feature_file = args.train_data_file
    
    ccdata = pd.read_pickle(codechange_file)
    fedata = pd.read_pickle(feature_file)
    
    if args.dataset_name:
        target_data = fedata[fedata['project'] == args.dataset_name]

        if args.cross_project:    
            #For updating skewed oversampling parameters, consider only data from the target project.
            fedata = target_data
            commits = []
            labels = []
            commit_messages = []
            codes = []

            for _, row in fedata.iterrows():
                commit_id = row['commit_hash']
                idx = ccdata[0].index(commit_id)
                label = row['is_buggy_commit']
                commit_message = ccdata[2][idx]
                code = ccdata[3][idx]
                commits.append(commit_id)
                labels.append(label)
                commit_messages.append(commit_message)
                codes.append(code)

            assert len(commits) == len(labels) == len(commit_messages) == len(codes) == fedata.shape[0]
            ccdata = (commits, labels, commit_messages, codes)
        else:
            assert len(target_data) == len(fedata)

    ma_examples = process_fedata_ccdata(fedata[-args.window_size:], ccdata, ccdata_index=-args.window_size)

    return ma_examples

def process_fedata_ccdata(fedata, ccdata, ccdata_index=0):
    # store parsed data.
    examples = []

    # parse fedata.
    manual_features_columns = ["la", "ld", "nf", "ns", "nd", "entropy", "ndev",
                               "lt", "nuc", "age", "exp", "rexp", "sexp", "fix"]
    fedata = normalize_df(fedata, manual_features_columns)
    # standardize fedata along any features.
    manual_features = preprocessing.scale(fedata[manual_features_columns].to_numpy())
    fedata[manual_features_columns] = manual_features

    # parse ccdata.
    commit_ids, labels, msgs, codes = ccdata
    for commit_id, label, msg, code in zip(commit_ids[ccdata_index:], labels[ccdata_index:], msgs[ccdata_index:], codes[ccdata_index:]):
        manual_features = fedata[fedata["commit_hash"] == commit_id][manual_features_columns].to_numpy().squeeze()
        examples.append((commit_id, label, msg, code, manual_features))
    
    return examples

def parse_data_file_with_timestamp(args, mode):
    codechange_file = ""
    feature_file = ""
    if mode == "train":
        codechange_file, feature_file = args.train_data_file
    elif mode == "eval":
        codechange_file, feature_file = args.eval_data_file
    elif mode == "test":
        codechange_file, feature_file = args.test_data_file

    ccdata = pd.read_pickle(codechange_file)
    fedata = pd.read_pickle(feature_file)
    
    original_fedata = fedata.copy()

    # store parsed data.
    examples = []

    # parse fedata.
    manual_features_columns = ["la", "ld", "nf", "ns", "nd", "entropy", "ndev",
                               "lt", "nuc", "age", "exp", "rexp", "sexp", "fix"]
    fedata = normalize_df(fedata, manual_features_columns)
    # standardize fedata along any features.
    manual_features = preprocessing.scale(fedata[manual_features_columns].to_numpy())
    fedata[manual_features_columns] = manual_features

    # parse ccdata.
    commit_ids, labels, msgs, codes = ccdata
    for commit_id, label, msg, code in zip(commit_ids, labels, msgs, codes):
        manual_features = fedata[fedata["commit_hash"] == commit_id][manual_features_columns].to_numpy().squeeze()
        timestamp = original_fedata[original_fedata["commit_hash"] == commit_id]["author_date_unix_timestamp"].to_numpy().squeeze()
        examples.append((commit_id, label, msg, code, manual_features, int(timestamp)
                         ))

    if mode == "train":
        random.seed(args.seed)
        random.shuffle(examples)

    return examples

def further_parse(example, tokenizer, args):
    commit_id, label, msg, code, manual_features = example
    label = int(label)
    added_tokens = []
    removed_tokens = []
    msg_tokens = tokenizer.tokenize(msg)
    msg_tokens = msg_tokens[:min(args.max_msg_length, len(msg_tokens))]

    added_codes = [' '.join(line.split()) for line in code['added_code']]
    codes = '[ADD]'.join([line for line in added_codes if len(line)])
    added_tokens.extend(tokenizer.tokenize(codes))

    removed_codes = [' '.join(line.split()) for line in code['removed_code']]
    codes = '[DEL]'.join([line for line in removed_codes if len(line)])
    removed_tokens.extend(tokenizer.tokenize(codes))

    input_tokens = msg_tokens + ['[ADD]'] + added_tokens + ['[DEL]'] + removed_tokens
    # Step 1 adaptation for CodeT5+: handle absence of cls_token / sep_token
    max_core = args.max_input_tokens - 2  # default reserve for special tokens
    # Determine boundary tokens gracefully
    cls_tok = getattr(tokenizer, 'cls_token', None)
    sep_tok = getattr(tokenizer, 'sep_token', None)
    eos_tok = getattr(tokenizer, 'eos_token', None)
    bos_tok = getattr(tokenizer, 'bos_token', None)
    # For T5 family usually only eos_token is defined; no cls_token
    if cls_tok is None:
        # Use bos if available, else reuse eos, else fallback to first manual special
        cls_tok = bos_tok or eos_tok or '<s>'
    if sep_tok is None:
        sep_tok = eos_tok or '</s>'
    input_tokens = input_tokens[:max_core]
    input_tokens = [cls_tok] + input_tokens + [sep_tok]
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    input_mask = [1] * len(input_ids)

    padding_length = args.max_input_tokens - len(input_ids)
    input_ids = input_ids + ([0] * padding_length)
    input_mask = input_mask + ([0] * padding_length)

    assert len(input_ids) == args.max_input_tokens
    assert len(input_mask) == args.max_input_tokens

    return commit_id, torch.tensor(input_ids), torch.tensor(input_mask), torch.tensor(manual_features), label

def further_parse_with_text_manual_features(example, tokenizer, args):
    commit_id, label, msg, code, manual_features = example
    label = int(label)
    added_tokens = []
    removed_tokens = []

    manual_features_columns = ["la", "ld", "nf", "ns", "nd", "entropy", "ndev",
                               "lt", "nuc", "age", "exp", "rexp", "sexp", "fix"]
    manual_features_tokens = [f"{manual_features_columns[i]}: {manual_features[i]:.3f}" for i, _ in enumerate(manual_features_columns)]
    manual_features_str = ', '.join([i for i in manual_features_tokens])
    manual_features_tokens = tokenizer.tokenize(manual_features_str)

    msg_tokens = tokenizer.tokenize(msg)
    msg_tokens = msg_tokens[:min(args.max_msg_length, len(msg_tokens))]

    added_codes = [' '.join(line.split()) for line in code['added_code']]
    codes = '[ADD]'.join([line for line in added_codes if len(line)])
    added_tokens.extend(tokenizer.tokenize(codes))

    removed_codes = [' '.join(line.split()) for line in code['removed_code']]
    codes = '[DEL]'.join([line for line in removed_codes if len(line)])
    removed_tokens.extend(tokenizer.tokenize(codes))

    input_tokens = ["<s>"] + manual_features_tokens + ["<s>"] + msg_tokens + ['[ADD]'] + added_tokens + ['[DEL]'] + removed_tokens
    max_core = args.max_input_tokens - 2
    cls_tok = getattr(tokenizer, 'cls_token', None) or getattr(tokenizer, 'bos_token', None) or getattr(tokenizer, 'eos_token', None) or '<s>'
    sep_tok = getattr(tokenizer, 'sep_token', None) or getattr(tokenizer, 'eos_token', None) or '</s>'
    input_tokens = input_tokens[:max_core]
    input_tokens = [cls_tok] + input_tokens + [sep_tok]
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    input_mask = [1] * len(input_ids)

    padding_length = args.max_input_tokens - len(input_ids)
    input_ids = input_ids + ([0] * padding_length)
    input_mask = input_mask + ([0] * padding_length)

    assert len(input_ids) == args.max_input_tokens
    assert len(input_mask) == args.max_input_tokens

    return torch.tensor(input_ids), torch.tensor(input_mask), torch.tensor(manual_features), label

def further_parse_with_timestamp(example, tokenizer, args):
    commit_id, label, msg, code, manual_features, timestamp = example
    label = int(label)
    added_tokens = []
    removed_tokens = []
    msg_tokens = tokenizer.tokenize(msg)
    msg_tokens = msg_tokens[:min(args.max_msg_length, len(msg_tokens))]

    added_codes = [' '.join(line.split()) for line in code['added_code']]
    codes = '[ADD]'.join([line for line in added_codes if len(line)])
    added_tokens.extend(tokenizer.tokenize(codes))

    removed_codes = [' '.join(line.split()) for line in code['removed_code']]
    codes = '[DEL]'.join([line for line in removed_codes if len(line)])
    removed_tokens.extend(tokenizer.tokenize(codes))

    input_tokens = msg_tokens + ['[ADD]'] + added_tokens + ['[DEL]'] + removed_tokens
    max_core = args.max_input_tokens - 2
    cls_tok = getattr(tokenizer, 'cls_token', None) or getattr(tokenizer, 'bos_token', None) or getattr(tokenizer, 'eos_token', None) or '<s>'
    sep_tok = getattr(tokenizer, 'sep_token', None) or getattr(tokenizer, 'eos_token', None) or '</s>'
    input_tokens = input_tokens[:max_core]
    input_tokens = [cls_tok] + input_tokens + [sep_tok]
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    input_mask = [1] * len(input_ids)

    padding_length = args.max_input_tokens - len(input_ids)
    input_ids = input_ids + ([0] * padding_length)
    input_mask = input_mask + ([0] * padding_length)

    assert len(input_ids) == args.max_input_tokens
    assert len(input_mask) == args.max_input_tokens

    return torch.tensor(input_ids), torch.tensor(input_mask), torch.tensor(manual_features), label, timestamp

def further_parse_msg_manual(example, tokenizer, args):
    _, label, msg, _, manual_features = example
    label = int(label)
    msg_tokens = tokenizer.tokenize(msg)
    msg_tokens = msg_tokens[:min(args.max_msg_length, len(msg_tokens))]

    input_tokens = msg_tokens
    max_core = args.max_input_tokens - 2
    cls_tok = getattr(tokenizer, 'cls_token', None) or getattr(tokenizer, 'bos_token', None) or getattr(tokenizer, 'eos_token', None) or '<s>'
    sep_tok = getattr(tokenizer, 'sep_token', None) or getattr(tokenizer, 'eos_token', None) or '</s>'
    input_tokens = input_tokens[:max_core]
    input_tokens = [cls_tok] + input_tokens + [sep_tok]
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    input_mask = [1] * len(input_ids)

    padding_length = args.max_input_tokens - len(input_ids)
    input_ids = input_ids + ([0] * padding_length)
    input_mask = input_mask + ([0] * padding_length)

    assert len(input_ids) == args.max_input_tokens
    assert len(input_mask) == args.max_input_tokens

    return torch.tensor(input_ids), torch.tensor(input_mask), torch.tensor(manual_features), label

def further_parse_manual(example):
    _, label, _, _, manual_features = example
    label = int(label)
    return torch.tensor([]), torch.tensor([]), torch.tensor(manual_features), label


class JITFineDataset(Dataset):
    def __init__(self, tokenizer, args, mode, is_ma_dataset=False):
        if is_ma_dataset:
            self.mid_examples = parse_ma_data(args)
        else:
            self.mid_examples = parse_data_file(args, mode)
        
        self.examples = [further_parse(item, tokenizer, args) for item in self.mid_examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]
    
def preprocess_code_line(code, remove_python_common_tokens=False):
    code = code.replace('(', ' ').replace(')', ' ').replace('{', ' ').replace('}', ' ').replace('[', ' ').replace(']',
                                                                                                                  ' ').replace(
        '.', ' ').replace(':', ' ').replace(';', ' ').replace(',', ' ').replace(' _ ', '_')

    code = re.sub('``.*``', '<STR>', code)
    code = re.sub("'.*'", '<STR>', code)
    code = re.sub('".*"', '<STR>', code)
    code = re.sub('\d+', '<NUM>', code)

    code = code.split()
    code = ' '.join(code)
    if remove_python_common_tokens:
        new_code = ''
        python_common_tokens = []
        for tok in code.split():
            if tok not in [python_common_tokens]:
                new_code = new_code + tok + ' '

        return new_code.strip()

    else:
        return code.strip()
    
class JITFineDatasetWithTextManualFeatures(Dataset):
    def __init__(self, tokenizer, args, mode):
        self.mid_examples = parse_data_file(args, mode)
        self.examples = [further_parse_with_text_manual_features(item, tokenizer, args) for item in self.mid_examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

class JITFineManualDataset(Dataset):
    def __init__(self, args, mode):
        self.mid_examples = parse_data_file(args, mode)
        self.examples = [further_parse_manual(item) for item in self.mid_examples]
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

class JITFineMessageManualDataset(Dataset):
    def __init__(self, tokenizer, args, mode):
        self.mid_examples = parse_data_file(args, mode)
        self.examples = [further_parse_msg_manual(item, tokenizer, args) for item in self.mid_examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]
    
if __name__ == "__main__":
    args = parse_jit_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = len(args.available_gpu)
    args.device = device

    set_seed(args)

    model, tokenizer, config = build_model_tokenizer_config(args)

    examples = parse_data_file(args, "test")
    print(examples[0])

    # train_dataset = JITFineDataset(tokenizer, args, "train")
    # eval_dataset = JITFineDataset(tokenizer, args, "eval")
    # test_dataset = JITFineDataset(tokenizer, args, "test")
    # print(len(train_dataset))
    # print(len(eval_dataset))
    # print(len(test_dataset))


