import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_line_level_metrics(line_score, label):
    scaler = MinMaxScaler()
    line_score = scaler.fit_transform(np.array(line_score).reshape(-1, 1))  # cannot pass line_score as list T-T
    pred = np.round(line_score)

    line_df = pd.DataFrame()
    line_df['scr'] = [float(val) for val in list(line_score)]
    line_df['label'] = label
    line_df = line_df.sort_values(by='scr', ascending=False)
    line_df['row'] = np.arange(1, len(line_df) + 1)

    line_df.to_csv('line_df.csv')

    real_buggy_lines = line_df[line_df['label'] == 1]

    top_10_acc = 0
    top_5_acc = 0

    if len(real_buggy_lines) < 1:
        IFA = len(line_df)
        top_20_percent_LOC_recall = 0
        effort_at_20_percent_LOC_recall = math.ceil(0.2 * len(line_df))

    else:
        IFA = line_df[line_df['label'] == 1].iloc[0]['row'] - 1
        label_list = list(line_df['label'])

        all_rows = len(label_list)

        # find top-10 accuracy
        if all_rows < 10:
            top_10_acc = np.sum(label_list[:all_rows]) / len(label_list[:all_rows])
        else:
            top_10_acc = np.sum(label_list[:10]) / len(label_list[:10])

        # find top-5 accuracy
        if all_rows < 5:
            top_5_acc = np.sum(label_list[:all_rows]) / len(label_list[:all_rows])
        else:
            top_5_acc = np.sum(label_list[:5]) / len(label_list[:5])

        #################### Adjustment: considers 1 if at least one bug was found
        if top_5_acc > 0:
            top_5_acc = 1.0
        
        if top_10_acc > 0:
            top_10_acc = 1.0
        ###########################################################################

        # find recall
        LOC_20_percent = line_df.head(int(0.2 * len(line_df)))
        buggy_line_num = LOC_20_percent[LOC_20_percent['label'] == 1]
        top_20_percent_LOC_recall = float(len(buggy_line_num)) / float(len(real_buggy_lines))

        # find effort @20% LOC recall

        buggy_20_percent = real_buggy_lines.head(math.ceil(0.2 * len(real_buggy_lines)))
        buggy_20_percent_row_num = buggy_20_percent.iloc[-1]['row']
        effort_at_20_percent_LOC_recall = int(buggy_20_percent_row_num) / float(len(line_df))

    return IFA, top_20_percent_LOC_recall, effort_at_20_percent_LOC_recall, top_10_acc, top_5_acc

def find_positions(input_tokens, token_id):
    if isinstance(input_tokens, np.ndarray):
        t = torch.from_numpy(input_tokens)
    elif torch.is_tensor(input_tokens):
        t = input_tokens
    else:
        t = torch.tensor(input_tokens)

    token_id = int(token_id)

    idx = (t == token_id).nonzero(as_tuple=False).squeeze(-1)  
    return idx  

def deal_with_attns(tokenizer, add_token_id, del_token_id, input_tokens, attns, commit2codes, idx2label,
                    only_adds=False, attn_temp: float = 1.0, deleted_weight: float = 0.75):
    '''
    score for each token
    :param item:
    :param attns:
    :param pred:
    :param commit2codes:
    :param idx2label:
    :return:
    '''

    # remove msg,cls,eos,del
    #begin_pos = input_tokens.index('[ADD]')
    #begin_pos = torch.where(input_tokens == add_token_id)[0]
    begin_pos_tensor = find_positions(input_tokens, add_token_id)
    begin_pos = begin_pos_tensor[0].item() if begin_pos_tensor.numel() > 0 else -1
    end_pos_tensor = find_positions(input_tokens, del_token_id)
    end_pos = end_pos_tensor[0].item() if end_pos_tensor.numel() > 0 else -1

    # --- Correction 1: Robust handling of missing or invalid [ADD]/[DEL] positions ---
    # If [ADD] not found, we cannot compute meaningful line-level attention -> skip (neutral metrics)
    if begin_pos == -1:
        logging.warning("[deal_with_attns] '[ADD]' token not found in input sequence; skipping commit for localization metrics.")
        return 0, 0, 0, 0, 0
    # If [DEL] not found or appears before [ADD], assume code diff extends to end of sequence
    if end_pos == -1 or end_pos <= begin_pos:
        end_pos = len(input_tokens)

    begin_pos = int(begin_pos) if torch.is_tensor(begin_pos) else begin_pos
    end_pos = int(end_pos) if torch.is_tensor(end_pos) else end_pos

    span_ids = input_tokens[begin_pos:]
    if (isinstance(span_ids, torch.Tensor) and span_ids.numel() == 0) or (hasattr(span_ids, '__len__') and len(span_ids) == 0):
        logging.warning("[deal_with_attns] Empty span after begin_pos=%s; returning neutral metrics.", begin_pos)
        return 0, 0, 0, 0, 0
    if torch.is_tensor(span_ids):
        span_ids = span_ids.detach().cpu().tolist()

    attn_vec = attns.mean(axis=0)
    attn_slice = attn_vec[begin_pos:]
    if isinstance(attn_slice, torch.Tensor):
        attn_slice_np = attn_slice.detach().cpu().numpy()
    else:
        attn_slice_np = np.asarray(attn_slice, dtype=np.float32)

    denom = attn_slice_np.sum()
    if denom <= 0:
        attn_slice_np = np.ones_like(attn_slice_np) / len(attn_slice_np)
    else:
        attn_slice_np = attn_slice_np / denom

    temp = float(attn_temp) if attn_temp is not None else 1.0
    if temp <= 0:
        temp = 1.0
    if abs(temp - 1.0) > 1e-6:
        safe = np.clip(attn_slice_np, 1e-12, 1.0)
        logits = np.log(safe)
        scaled = logits / temp
        scaled = scaled - scaled.max()
        expv = np.exp(scaled)
        attn_slice_np = expv / expv.sum()

    attn_records = []
    seen_deleted = False
    for offset, token_id in enumerate(span_ids):
        if token_id == add_token_id:
            continue
        if token_id == del_token_id:
            seen_deleted = True
            continue
        token = tokenizer.convert_ids_to_tokens([token_id])[0]
        norm_token = token.replace('\u0120', '')
        segment = 'deleted' if seen_deleted else 'added'
        attn_records.append({
            'token': norm_token,
            'segment': segment,
            'score': float(attn_slice_np[offset])
        })

    if not attn_records:
        logging.warning("[deal_with_attns] No diff tokens collected after processing attention; returning neutral metrics.")
        return 0, 0, 0, 0, 0

    attn_df = pd.DataFrame(attn_records)
    if only_adds:
        attn_df = attn_df[attn_df['segment'] == 'added']
    if attn_df.empty:
        logging.warning("[deal_with_attns] Attention dataframe empty after applying only_adds=%s", only_adds)
        return 0, 0, 0, 0, 0

    # calculate score for each line in commit
    def _normalize_change_type(value: str) -> str:
        value = value.lower()
        if 'add' in value:
            return 'added'
        if 'del' in value or 'remov' in value:
            return 'deleted'
        return value

    commit2codes = commit2codes.copy()
    commit2codes['changed_type'] = commit2codes['changed_type'].astype(str).map(_normalize_change_type)
    attn_df['segment'] = attn_df['segment'].astype(str).map(_normalize_change_type)

    if only_adds:
        commit2codes = commit2codes[commit2codes['changed_type'] == 'added']
        attn_df = attn_df[attn_df['segment'] == 'added']

    merge_keys_left = ['token', 'changed_type']
    merge_keys_right = ['token', 'segment']

    if commit2codes.empty or attn_df.empty:
        logging.warning("[deal_with_attns] Empty commit2codes or attn_df after filtering; returning neutral metrics.")
        return 0, 0, 0, 0, 0

    commit2codes = commit2codes.drop('commit_id', axis=1)

    result_df = pd.merge(commit2codes, attn_df, how='left', left_on=merge_keys_left, right_on=merge_keys_right)
    result_df = result_df.drop(columns=['segment'])
    result_df['score'] = result_df['score'].fillna(0.0)
    result_df['token_count'] = 1

    if not only_adds and deleted_weight is not None:
        try:
            dw = max(0.0, float(deleted_weight))
        except (TypeError, ValueError):
            dw = 0.75
        deleted_mask = result_df['changed_type'] == 'deleted'
        result_df.loc[deleted_mask, 'score'] *= dw

    line_scores = result_df.groupby('idx').agg(
        line_score=('score', 'sum'),
        token_count=('token_count', 'sum')
    ).reset_index()

    line_scores['score'] = line_scores['line_score'] / np.sqrt(np.clip(line_scores['token_count'], a_min=1, a_max=None))
    result_df = line_scores.drop(columns=['line_score'])

    result_df.to_csv('result_df.csv')

    result_df = pd.merge(result_df, idx2label, how='inner', on='idx')
    IFA, top_20_percent_LOC_recall, effort_at_20_percent_LOC_recall, top_10_acc, top_5_acc = get_line_level_metrics(
        result_df['score'].tolist(), result_df['label'].tolist())
    return IFA, top_20_percent_LOC_recall, effort_at_20_percent_LOC_recall, top_10_acc, top_5_acc

def commit_with_codes(filepath, tokenizer):
    data = pd.read_pickle(filepath)
    commit2codes = []
    idx2label = []
    for _, item in data.iterrows():
        commit_id, idx, changed_type, label, _, changed_line = item
        line_tokens = [token.replace('\u0120', '') for token in tokenizer.tokenize(changed_line)]
        for token in line_tokens:
            commit2codes.append([commit_id, idx, changed_type, token])
        idx2label.append([commit_id, idx, label])
    commit2codes = pd.DataFrame(commit2codes, columns=['commit_id', 'idx', 'changed_type', 'token'])
    idx2label = pd.DataFrame(idx2label, columns=['commit_id', 'idx', 'label'])
    return commit2codes, idx2label


def locate_defects(tokenizer, test_dataset, y_preds, probs, attns, args):
    result = []

    cache_buggy_line = os.path.join(os.path.dirname(args.buggy_line_filepath),
                                    'changes_complete_buggy_line_level_cache.pkl')
    if os.path.exists(cache_buggy_line):
        commit2codes, idx2label = pickle.load(open(cache_buggy_line, 'rb'))
    else:
        commit2codes, idx2label = commit_with_codes(args.buggy_line_filepath, tokenizer)
        pickle.dump((commit2codes, idx2label), open(cache_buggy_line, 'wb'))

    IFA, top_20_percent_LOC_recall, effort_at_20_percent_LOC_recall, top_10_acc, top_5_acc = [], [], [], [], []
    for example, pred, prob, attn in zip(test_dataset.examples, y_preds, probs[:, -1], attns):
        commit_id = example[0]
        input_tokens = example[1]
        label = example[-1]
        result.append([commit_id, prob, pred, label])

        # calculate
        add_token_id = tokenizer.convert_tokens_to_ids('[ADD]')
        del_token_id = tokenizer.convert_tokens_to_ids('[DEL]')
        if int(label) == 1 and int(pred) == 1 and add_token_id in input_tokens:
            cur_codes = commit2codes[commit2codes['commit_id'] == commit_id]
            cur_labels = idx2label[idx2label['commit_id'] == commit_id]
            cur_IFA, cur_top_20_percent_LOC_recall, cur_effort_at_20_percent_LOC_recall, cur_top_10_acc, cur_top_5_acc = deal_with_attns(
                tokenizer, add_token_id, del_token_id, input_tokens, attn, cur_codes, cur_labels,
                args.only_adds, getattr(args, 'loc_attn_temp', 1.0), getattr(args, 'loc_deleted_weight', 0.75))
            IFA.append(cur_IFA)
            top_20_percent_LOC_recall.append(cur_top_20_percent_LOC_recall)
            effort_at_20_percent_LOC_recall.append(cur_effort_at_20_percent_LOC_recall)
            top_10_acc.append(cur_top_10_acc)
            top_5_acc.append(cur_top_5_acc)

    logger.info(
        'Top-10-ACC: {:.4f},Top-5-ACC: {:.4f}, Recall20%Effort: {:.4f}, Effort@20%LOC: {:.4f}, IFA: {:.4f}'.format(
            round(np.mean(top_10_acc), 4), round(np.mean(top_5_acc), 4),
            round(np.mean(top_20_percent_LOC_recall), 4),
            round(np.mean(effort_at_20_percent_LOC_recall), 4), round(np.mean(IFA), 4))
    )
    print(
        'Top-10-ACC: {:.4f},Top-5-ACC: {:.4f}, Recall20%Effort: {:.4f}, Effort@20%LOC: {:.4f}, IFA: {:.4f}'.format(
            round(np.mean(top_10_acc), 4), round(np.mean(top_5_acc), 4),
            round(np.mean(top_20_percent_LOC_recall), 4),
            round(np.mean(effort_at_20_percent_LOC_recall), 4), round(np.mean(IFA), 4))
    )
    RF_result = pd.DataFrame(result)
    RF_result.to_csv(os.path.join(args.output_dir, "predictions.csv"), sep='\t', index=None)