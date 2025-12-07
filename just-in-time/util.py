import argparse
import random
import numpy as np
import torch
from transformers import (RobertaModel, RobertaTokenizer, RobertaConfig, T5ForConditionalGeneration, T5Config,
                          PLBartTokenizer, PLBartForConditionalGeneration, PLBartConfig, CodeGenTokenizer, 
                          AutoTokenizer, AutoModel)
import logging


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"yes", "true", "t", "1", "y"}:
        return True
    if value in {"no", "false", "f", "0", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_jit_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data_file", nargs=2, type=str)
    parser.add_argument("--eval_data_file", nargs=2, type=str)
    parser.add_argument("--test_data_file", nargs=2, type=str)
    parser.add_argument("--output_dir", type=str, default=None)

    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--pretrained_model", type=str, default="plbart")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=5,
                        help='patience for early stop')

    parser.add_argument("--manual_feature_size", type=int, default=14)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", type=str, default="tanh")

    parser.add_argument("--max_msg_length", type=int, default=64)
    parser.add_argument("--max_input_tokens", type=int, default=512)

    parser.add_argument("--available_gpu", type=list, default=[1, 2, 3, 0])
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--use_lora", action='store_true')

    parser.add_argument("--eval_metric", type=str, default="f1")
    parser.add_argument("--do_resume_training", action='store_true', default=False)
    parser.add_argument("--oversample", action='store_true', default=False)
    parser.add_argument("--undersample", action='store_true', default=False)
    parser.add_argument("--online_mode", action='store_true', default=False)
    parser.add_argument("--calculate_metrics", action='store_true', default=False)
    
    parser.add_argument("--skewed_oversample", action='store_true', default=False)
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--target_th", type=float, default=0.5)
    parser.add_argument("--l0", type=float, default=10)
    parser.add_argument("--l1", type=float, default=12)
    parser.add_argument("--m", type=float, default=1.5)
    parser.add_argument("--only_manual", action='store_true', default=False)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--cross_project", action='store_true', default=False)
    parser.add_argument("--buggy_line_filepath", type=str, default=None)
    parser.add_argument("--only_adds", type=str2bool, nargs='?', const=True, default=True,
                        help="Whether to consider only added lines for localization (pass False to include deletions)")

    # Localization inference-only enhancements
    parser.add_argument("--loc_last_k_layers", type=int, default=1, help="Number of last encoder layers to average attentions over for defect localization (1 = last layer only).")
    parser.add_argument("--loc_layer_weighting", type=str, default="none", choices=["none", "exp"], help="Strategy to combine last k layers: uniform average or exponential weights (exp higher weight to last).")
    parser.add_argument("--loc_layer_exp_alpha", type=float, default=0.7, help="Alpha for exponential layer weighting: w_i = exp(-alpha*(k-1-i)). Higher alpha decays older layers faster.")
    parser.add_argument("--loc_attn_temp", type=float, default=1.0, help="Temperature (<1 sharpens, >1 suaviza) aplicada à distribuição de atenção CLS->tokens antes da agregação por linha.")
    parser.add_argument("--loc_deleted_weight", type=float, default=0.75,
                        help="Fator multiplicativo para scores de linhas deletadas quando only_adds=False.")
    
    #Localization enhancements
    parser.add_argument('--max_codeline_length', type=int, default=256,
                        help="max_codeline_length")
    parser.add_argument('--max_codeline_token_length', type=int, default=64,
                        help="max_codeline_token_length")
    parser.add_argument("--buggy_lines_file", nargs=3, type=str, required=False,
                        help="The input xxx_buggy_commit_lines_df.pkl (a .pkl file).")
    parser.add_argument('--dp_loss_weight', type=float, default=0.3,
                        help="The loss weight of the defect prediction task")
    parser.add_argument('--dl_loss_weight', type=float, default=0.7,
                        help="The loss weight of the defect localization task")
    parser.add_argument("--do_locate_defects", action='store_true', default=False,
                        help="Whether to perform defect localization in addition to defect prediction.")

    args = parser.parse_args()
    return args


def set_seed(args):
    print("Seed = ", args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def build_model_tokenizer_config(args):
    model_classes = {
        "codebert": (RobertaModel, RobertaTokenizer, RobertaConfig, "microsoft/codebert-base"),
        "graphcodebert": (RobertaModel, RobertaTokenizer, RobertaConfig, "microsoft/graphcodebert-base"),
        "codet5": (T5ForConditionalGeneration, RobertaTokenizer, T5Config, "Salesforce/codet5-base"),
        "unixcoder": (RobertaModel, RobertaTokenizer, RobertaConfig, "microsoft/unixcoder-base"),
        "plbart": (PLBartForConditionalGeneration, PLBartTokenizer, PLBartConfig, "uclanlp/plbart-base"),
        "plbart-large": (PLBartForConditionalGeneration, PLBartTokenizer, PLBartConfig, "uclanlp/plbart-large"),
        #fork by monilouise
        "codet5p": (T5ForConditionalGeneration, RobertaTokenizer, T5Config, "Salesforce/codet5p-220m-bimodal"),
        "codet5p-770m": (T5ForConditionalGeneration, RobertaTokenizer, T5Config, "Salesforce/codet5p-770m"),
        "codet5p-2b": (T5ForConditionalGeneration, CodeGenTokenizer, T5Config, "Salesforce/codet5p-2b"),
        "codet5p-6b": (T5ForConditionalGeneration, CodeGenTokenizer, T5Config, "Salesforce/codet5p-6b"),
        "codet5p-16b": (T5ForConditionalGeneration, CodeGenTokenizer, T5Config, "Salesforce/codet5p-16b"),
        "codereviewer": (T5ForConditionalGeneration, RobertaTokenizer, T5Config, "microsoft/codereviewer"),
        "modernbert": (AutoModel, AutoTokenizer, None, "answerdotai/ModernBERT-base"),
        "modernbert-large": (AutoModel, AutoTokenizer, None, "answerdotai/ModernBERT-large"),
    }

    model_class, tokenizer_class, config_class, actual_name = model_classes[args.pretrained_model]

    # load config.
    config = None
    if config_class:
        config = config_class.from_pretrained(actual_name)
        if args.pretrained_model in ["codebert", "graphcodebert", "unixcoder"]:
            config.hidden_size = args.hidden_size
        elif args.pretrained_model in ["codet5", "plbart", "plbart-large"]:
            config.d_model = args.hidden_size

        config.hidden_dropout_prob = args.dropout
        config.attention_probs_dropout_prob = args.dropout
    
    # load tokenizer.
    tokenizer = tokenizer_class.from_pretrained(actual_name)
    special_tokens_dict = {"additional_special_tokens": ["[ADD]", "[DEL]"]}

    if args.pretrained_model in ["codet5p-2b", "codet5p-6b", "codet5p-16b"]:
        special_tokens_dict["additional_special_tokens"].extend(["<s>", "</s>"])
        tokenizer.cls_token = "<s>"
        tokenizer.sep_token = "</s>"

    tokenizer.add_special_tokens(special_tokens_dict)
    # load pretrained model.
    if config:
        model = model_class.from_pretrained(actual_name, config=config)
    else:
        logger = logging.getLogger(__name__)
        logger.info(f"Loading model {actual_name}")
        model = model_class.from_pretrained(actual_name)

    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer, config


