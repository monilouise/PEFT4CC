import torch
import torch.nn as nn
from torchvision.ops.focal_loss import sigmoid_focal_loss
import torch.nn.functional as F

class RobertaClassifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.manual_dense = nn.Linear(args.manual_feature_size, args.hidden_size)
        self.dropout = nn.Dropout(args.dropout)
        self.cat_proj = nn.Linear(2 * args.hidden_size, 1)

    def forward(self, features, manual_features):
        cls_features = features[:, 0, :]
        manual_features = manual_features.float()
        manual_features = self.manual_dense(manual_features)

        if self.args.activation == "tanh":
            manual_features = torch.tanh(manual_features)
        elif self.args.activation == "relu":
            manual_features = torch.relu(manual_features)

        cat_features = torch.cat((cls_features, manual_features), dim=1)

        cat_features = self.dropout(cat_features)
        proj_score = self.cat_proj(cat_features)
        return proj_score


class ConcatModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(ConcatModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.classifier = RobertaClassifier(args)
        # Logging stats (Correction 3 instrumentation)
        self._attn_success = 0
        self._attn_fail = 0

    def forward(self, input_ids, input_mask, manual_features, label=None, output_attentions=None):
        if self.args.pretrained_model in ["codebert", "graphcodebert", "unixcoder", "modernbert", "modernbert-large"]:
            if self.args.use_lora:
                outputs = self.encoder.base_model.model(input_ids=input_ids, attention_mask=input_mask, output_attentions=output_attentions)
            else:
                outputs = self.encoder(input_ids=input_ids, attention_mask=input_mask, 
                                       output_attentions=output_attentions)
        elif self.args.pretrained_model in ["codet5", "codet5p", "codet5p-770m", "codet5p-2b", 
                                            "codet5p-6b", "codet5p-16b", "codereviewer"]:
            outputs = self.encoder.encoder(input_ids=input_ids, attention_mask=input_mask, 
                                           output_attentions=output_attentions)
        elif self.args.pretrained_model in ["plbart", "plbart-large"]:
            if self.args.use_lora:
                outputs = self.encoder.base_model.model.model.encoder(input_ids=input_ids, 
                                                                      attention_mask=input_mask, 
                                                                      output_attentions=output_attentions)
            else:
                outputs = self.encoder.model.encoder(input_ids=input_ids, attention_mask=input_mask, 
                                                     output_attentions=output_attentions)

        logits = self.classifier(outputs[0], manual_features)

        # Correction 3: robust extraction of last-layer attention (CLS -> tokens)
        last_layer_attn_weights = None
        if output_attentions and hasattr(outputs, 'attentions') and outputs.attentions is not None:
            try:
                # Prefer encoder.config over self.config (may be None for some AutoModels)
                num_layers = getattr(self.encoder, 'config', None)
                if num_layers is not None and hasattr(num_layers, 'num_hidden_layers'):
                    total_layers = num_layers.num_hidden_layers
                else:
                    total_layers = len(outputs.attentions)
                k = getattr(self.args, 'loc_last_k_layers', 1)
                if k < 1:
                    k = 1
                k = min(k, total_layers)
                # Stack last k layers: list[-k:] each (B,H,L,L)
                selected = outputs.attentions[total_layers - k: total_layers]  # list length k
                if isinstance(selected, (list, tuple)) and len(selected) > 0 and selected[0].dim() == 4:
                    stacked = torch.stack(selected, dim=0)  # (k,B,H,L,L)
                    weighting_mode = getattr(self.args, 'loc_layer_weighting', 'none')
                    if weighting_mode == 'exp' and k > 1:
                        alpha = getattr(self.args, 'loc_layer_exp_alpha', 0.7)
                        # indices: 0 oldest ... k-1 newest (want larger weights for newer)
                        idx = torch.arange(k, device=stacked.device)
                        # w_i = exp(-alpha*(k-1-i)) => largest when i=k-1
                        weights = torch.exp(-alpha * (k - 1 - idx))
                        weights = weights / weights.sum()
                        # reshape for broadcasting: (k,1,1,1,1)
                        weights = weights.view(k, 1, 1, 1, 1)
                        layer_attn = (stacked * weights).sum(dim=0)
                    else:
                        layer_attn = stacked.mean(dim=0)
                    last_layer_attn_weights = layer_attn[:, :, 0, :].detach()
                # else: leave None
            except Exception as e:
                # Silent fallback; logging can be added if desired
                last_layer_attn_weights = None

        #Original
        prob = torch.sigmoid(logits)

        if label is not None:
            loss_fct = nn.BCELoss()
            loss = loss_fct(prob, torch.unsqueeze(label, dim=1).float())
            # Update counters only during evaluation/inference when attention requested
            if output_attentions:
                if last_layer_attn_weights is not None:
                    self._attn_success += 1
                else:
                    self._attn_fail += 1
                    if (self._attn_fail + self._attn_success) % 50 == 0:
                        # Lazy import logging to avoid global dependency if not used
                        import logging
                        logging.getLogger(__name__).warning(
                            f"Attention extraction failures: {self._attn_fail} / "
                            f"{self._attn_fail + self._attn_success} batches")
            return prob, loss, last_layer_attn_weights
        else:
            if output_attentions:
                if last_layer_attn_weights is not None:
                    self._attn_success += 1
                else:
                    self._attn_fail += 1
                    if (self._attn_fail + self._attn_success) % 50 == 0:
                        import logging
                        logging.getLogger(__name__).warning(
                            f"Attention extraction failures: {self._attn_fail} / "
                            f"{self._attn_fail + self._attn_success} batches")
            return prob

    def copy(self):
        new_model = ConcatModel(self.encoder, self.config, self.tokenizer, self.args)
        new_model.load_state_dict(self.state_dict())
        new_model.to(self.args.device)
        return new_model
    
class Attention(nn.Module):       #x:[batch, seq_len, hidden_dim*2]

    def __init__(self, hidden_size, need_aggregation=True):
        super().__init__()
        self.need_aggregation = need_aggregation
        self.w = nn.Parameter(torch.Tensor(hidden_size * 2, hidden_size * 2))
        self.u = nn.Parameter(torch.Tensor(hidden_size * 2, 1))
        nn.init.uniform_(self.w, -0.1, 0.1)
        nn.init.uniform_(self.u, -0.1, 0.1)

    def forward(self, x):
        device = x.device
        self.w = self.w.to(device)
        self.u = self.u.to(device)

        u = torch.tanh(torch.matmul(x, self.w))         #[batch, seq_len, hidden_size*2]
        score = torch.matmul(u, self.u)                   #[batch, seq_len, 1]
        att = F.softmax(score, dim=1)

        scored_x = x * att                              #[batch, seq_len, hidden_size*2]

        if self.need_aggregation:
            context = torch.sum(scored_x, dim=1)                  #[batch, hidden_size*2]
            return context
        else:
            return scored_x
    