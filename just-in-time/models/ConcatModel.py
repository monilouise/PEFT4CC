import torch
import torch.nn as nn
from torchvision.ops.focal_loss import sigmoid_focal_loss

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

    def forward(self, input_ids, input_mask, manual_features, label):
        if self.args.pretrained_model in ["codebert", "graphcodebert", "unixcoder", "modernbert", "modernbert-large"]:
            if self.args.use_lora:
                outputs = self.encoder.base_model.model(input_ids=input_ids, attention_mask=input_mask)
            else:
                outputs = self.encoder(input_ids=input_ids, attention_mask=input_mask)
        elif self.args.pretrained_model in ["codet5", "codet5p", "codet5p-770m", "codet5p-2b", 
                                            "codet5p-6b", "codet5p-16b", "codereviewer"]:
            outputs = self.encoder.encoder(input_ids=input_ids, attention_mask=input_mask)
        elif self.args.pretrained_model in ["plbart", "plbart-large"]:
            if self.args.use_lora:
                outputs = self.encoder.base_model.model.model.encoder(input_ids=input_ids, attention_mask=input_mask)
            else:
                outputs = self.encoder.model.encoder(input_ids=input_ids, attention_mask=input_mask)

        logits = self.classifier(outputs[0], manual_features)

        #Original
        prob = torch.sigmoid(logits)
        loss_fct = nn.BCELoss()
        loss = loss_fct(prob, torch.unsqueeze(label, dim=1).float())
        
        #Focal loss 
        #prob = torch.sigmoid(logits)
        #loss = sigmoid_focal_loss(logits, torch.unsqueeze(label, dim=1).float(), gamma=2.0, reduction='mean')
        #loss = sigmoid_focal_loss(logits, torch.unsqueeze(label, dim=1).float(), alpha=0.9, gamma=0.0, reduction='mean')
        #loss = sigmoid_focal_loss(logits, torch.unsqueeze(label, dim=1).float(), alpha=0.75, gamma=0.0, reduction='mean')
        #loss = sigmoid_focal_loss(logits, torch.unsqueeze(label, dim=1).float(), alpha=0.5, gamma=0.0, reduction='mean')
        #loss = sigmoid_focal_loss(logits, torch.unsqueeze(label, dim=1).float(), alpha=0.99, gamma=0.0, reduction='mean')
        #loss = sigmoid_focal_loss(logits, torch.unsqueeze(label, dim=1).float(), alpha=0.999, gamma=0.0, reduction='mean')
        #loss = sigmoid_focal_loss(logits, torch.unsqueeze(label, dim=1).float(), alpha=0.1, gamma=0.0, reduction='mean')
        #loss = sigmoid_focal_loss(logits, torch.unsqueeze(label, dim=1).float(), alpha=0.75, gamma=0.1, reduction='mean')
        #loss = sigmoid_focal_loss(logits, torch.unsqueeze(label, dim=1).float(), alpha=0.75, gamma=0.2, reduction='mean')
        #loss = sigmoid_focal_loss(logits, torch.unsqueeze(label, dim=1).float(), alpha=0.5, gamma=0.5, reduction='mean')
        #loss = sigmoid_focal_loss(logits, torch.unsqueeze(label, dim=1).float(), alpha=0.25, gamma=1, reduction='mean')
        #loss = sigmoid_focal_loss(logits, torch.unsqueeze(label, dim=1).float(), alpha=0.25, gamma=5, reduction='mean')
        #loss = sigmoid_focal_loss(logits, torch.unsqueeze(label, dim=1).float(), alpha=0.25, gamma=0.0, reduction='mean')

        #End focal loss

        return prob, loss


    def copy(self):
        new_model = ConcatModel(self.encoder, self.config, self.tokenizer, self.args)
        new_model.load_state_dict(self.state_dict())
        new_model.to(self.args.device)
        return new_model

