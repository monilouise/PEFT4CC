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

    def forward(self, input_ids, input_mask, manual_features, label=None, output_attentions=None, attn_implementation=None):
        if self.args.pretrained_model in ["codebert", "graphcodebert", "unixcoder", "modernbert", "modernbert-large"]:
            if self.args.use_lora and not self.args.do_predict:
                outputs = self.encoder.base_model.model(input_ids=input_ids, attention_mask=input_mask, 
                                                        output_attentions=output_attentions, 
                                                        attn_implementation=attn_implementation)
            elif self.args.use_lora and self.args.do_predict: 
                outputs = self.encoder(input_ids=input_ids, attention_mask=input_mask, 
                                       output_attentions=output_attentions, attn_implementation=attn_implementation)
            else:
                outputs = self.encoder(input_ids=input_ids, attention_mask=input_mask, 
                                       output_attentions=output_attentions, attn_implementation=attn_implementation)
        elif self.args.pretrained_model in ["codet5", "codet5p", "codet5p-770m", "codet5p-2b", 
                                            "codet5p-6b", "codet5p-16b", "codereviewer"]:
            outputs = self.encoder.encoder(input_ids=input_ids, attention_mask=input_mask, 
                                           output_attentions=output_attentions, attn_implementation=attn_implementation)
        elif self.args.pretrained_model in ["plbart", "plbart-large"]:
            if self.args.use_lora:
                outputs = self.encoder.base_model.model.model.encoder(input_ids=input_ids, 
                                                                      attention_mask=input_mask, 
                                                                      output_attentions=output_attentions,
                                                                      attn_implementation=attn_implementation)
            else:
                outputs = self.encoder.model.encoder(input_ids=input_ids, attention_mask=input_mask, 
                                                     output_attentions=output_attentions, 
                                                     attn_implementation=attn_implementation)

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

class HAN_MODEL(nn.Module):
    def __init__(self, embedding_layer):
        super().__init__()
        self.hidden_size = 256
        self.num_layers = 1
        self.bidirectional = True
        self.embedding = embedding_layer

        self.lstm1 = nn.LSTM(input_size=self.embedding.embedding_dim,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            bidirectional=self.bidirectional,
                            batch_first=True)
        self.att1 = Attention(self.hidden_size, need_aggregation=True)
        self.lstm2 = nn.LSTM(input_size=self.hidden_size*2,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            bidirectional=self.bidirectional,
                            batch_first=True)
        self.att2 = Attention(self.hidden_size, need_aggregation=False)

        # self.fc1 = nn.Linear(512, 2)
        self.fc1 = nn.Linear(512, 128)
        self.relu = nn.Tanh()
        self.fc2 = nn.Linear(128, 2)
        # self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.5)


    def forward(self, x):
        # input x : (bs, nus_sentences, nums_words)
        device = x.device
        x = self.embedding(x) # out x : (bs, nus_sentences, nums_words, embedding_dim)
        x = self.dropout(x)
        batch_size, num_sentences, num_words, emb_dim = x.shape

        h0_1 = torch.randn(self.num_layers*2, batch_size*num_sentences, self.hidden_size).to(device)
        c0_1 = torch.randn(self.num_layers*2, batch_size*num_sentences, self.hidden_size).to(device)
        h0_2 = torch.randn(self.num_layers*2, batch_size, self.hidden_size).to(device)
        c0_2 = torch.randn(self.num_layers*2, batch_size, self.hidden_size).to(device)

        x = x.view(batch_size*num_sentences, num_words, emb_dim).contiguous()
        x,(_,_)= self.lstm1(x, (h0_1,c0_1))   # out：batch_size*num_sentences, num_words，hidden_size*2
        x = self.att1(x)   # batch_size*num_sentences, hidden_size*2

        x = x.view(x.size(0)//num_sentences, num_sentences, self.hidden_size*2).contiguous()
        x,(_,_)= self.lstm2(x, (h0_2,c0_2))   # out：batch_size, num_sentences，hidden_size*2

        x = self.att2(x)   # batch_size, hidden_size*2
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
    
class MultiFocalLoss(nn.Module):
    """
    Focal_Loss= -1*alpha*((1-pt)**gamma)*log(pt)
    Args:
        num_class: number of classes
        alpha: class balance factor shape=[num_class, ]
        gamma: hyper-parameter
        reduction: reduction type
    """

    def __init__(self, num_class, alpha=None, gamma=2, reduction='mean'):
        super(MultiFocalLoss, self).__init__()
        self.num_class = num_class
        self.gamma = gamma
        self.reduction = reduction
        self.smooth = 1e-4
        self.alpha = alpha
        if alpha is None:
            self.alpha = torch.ones(num_class, ) - 0.5
        elif isinstance(alpha, (int, float)):
            self.alpha = torch.as_tensor([alpha] * num_class)
        elif isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.as_tensor(alpha)
        if self.alpha.shape[0] != num_class:
            raise RuntimeError('the length not equal to number of class')

    def forward(self, logit, target):
        # assert isinstance(self.alpha,torch.Tensor)\
        alpha = self.alpha.to(logit.device)
        prob = F.softmax(logit, dim=1)

        if prob.dim() > 2:
            # used for 3d-conv:  N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            N, C = logit.shape[:2]
            prob = prob.view(N, C, -1)
            prob = prob.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
            prob = prob.view(-1, prob.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]

        ori_shp = target.shape
        target = target.view(-1, 1)

        prob = prob.gather(1, target).view(-1) + self.smooth  # avoid nan
        logpt = torch.log(prob)
        # alpha_class = alpha.gather(0, target.squeeze(-1))
        alpha_weight = alpha[target.squeeze().long()]
        loss = -alpha_weight * torch.pow(torch.sub(1.0, prob), self.gamma) * logpt

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'none':
            loss = loss.view(ori_shp)

        return loss

class ConcatModelDL(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(ConcatModelDL, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.classifier = RobertaClassifier(args)

        # ----------------------HAN-------------------------------
        #self.han_word_embedding_layer = self.encoder.embeddings.word_embeddings
        #self.han_word_embedding_layer = self.encoder.base_model.embeddings.word_embeddings
        self.han_word_embedding_layer = self.encoder.base_model.get_input_embeddings()
        self.han_locator = HAN_MODEL(embedding_layer=self.han_word_embedding_layer)

        # --------------------------------------------------------

    def forward(self, input_ids, input_mask, manual_features, label=None, line_ids=None, line_label=None, 
                output_attentions=None):
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

        last_layer_attn_weights = outputs.attentions[self.config.num_hidden_layers - 1][:, :,
                                  0].detach() if output_attentions else None 
        logits = self.classifier(outputs[0], manual_features)

        han_logits = self.han_locator(line_ids)
        
        #From JIT-SMART
        logits = (logits+han_logits.mean(dim=1))/2

        #Original
        #prob = torch.sigmoid(logits)
        prob = torch.sigmoid(logits[:, 1])  #prob of positive class

        prob_lines = torch.softmax(han_logits, dim=-1)[:, :, 1]

        if label is not None:
            loss_fct = nn.BCELoss()
            #loss1 = loss_fct(prob, torch.unsqueeze(label, dim=1).float())
            loss1 = loss_fct(prob, label.float())

            loss_fct = nn.BCELoss()
            
            #loss2 = loss_fct(prob_lines.view(-1), line_label.reshape((-1,)).float())
            loss_dl = MultiFocalLoss(alpha=0.25, gamma=2, reduction='mean', num_class=2)
            loss2 = loss_dl(han_logits.reshape((-1, 2)), line_label.reshape((-1,)))
            
            loss = loss1*self.args.dp_loss_weight + loss2*self.args.dl_loss_weight

            return prob, loss, last_layer_attn_weights, prob_lines 
        else:
            return prob, prob_lines

    def copy(self):
        new_model = ConcatModel(self.encoder, self.config, self.tokenizer, self.args)
        new_model.load_state_dict(self.state_dict())
        new_model.to(self.args.device)
        return new_model
