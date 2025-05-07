import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dropout = nn.Dropout(args.dropout)
        self.cat_proj = nn.Linear(args.manual_feature_size, 1)

    def forward(self, manual_features):
        manual_features = manual_features.float()
        
        if self.args.activation == "tanh":
            manual_features = torch.tanh(manual_features)
        elif self.args.activation == "relu":
            manual_features = torch.relu(manual_features)

        manual_features = self.dropout(manual_features)
        proj_score = self.cat_proj(manual_features)
        return proj_score


class ManualModel(nn.Module):
    def __init__(self, args):
        super(ManualModel, self).__init__()
        self.classifier = Classifier(args)

    def forward(self, input_ids, input_mask, manual_features, label):
        logits = self.classifier(manual_features)

        prob = torch.sigmoid(logits)

        loss_fct = nn.BCELoss()
        loss = loss_fct(prob, torch.unsqueeze(label, dim=1).float())

        return prob, loss

