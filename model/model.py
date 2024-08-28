import torch
import torch.nn as nn
from transformers import AutoModel

class NERPOSModel(nn.Module):
    def __init__(self, num_pos, num_ner):
        super(NERPOSModel, self).__init__()
        self.num_pos = num_pos
        self.num_ner = num_ner
        self.bert = AutoModel.from_pretrained("xlm-roberta-base")
        
        # Freeze all layers except the last 4 transformer layers
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.encoder.layer[-4:].parameters():
            param.requires_grad = True
        
        # Fully connected layers for POS and NER predictions
        self.fc_pos = nn.Linear(768, self.num_pos)
        self.fc_ner = nn.Linear(768, self.num_ner)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, ids, mask, token_type_ids=None, target_pos=None, target_ner=None):
        bert_out = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        
        # Apply dropout to the last hidden state
        dropout_out = self.dropout(bert_out.last_hidden_state)
        
        # Compute logits for POS and NER
        pos_logits = self.fc_pos(dropout_out)
        ner_logits = self.fc_ner(dropout_out)
        
        loss = None
        if target_pos is not None and target_ner is not None:
            criterion = nn.CrossEntropyLoss()
            active_loss = mask.view(-1) == 1
            active_logits_pos = pos_logits.view(-1, self.num_pos)
            active_logits_ner = ner_logits.view(-1, self.num_ner)
            active_labels_pos = torch.where(active_loss, target_pos.view(-1), torch.tensor(criterion.ignore_index).type_as(target_pos))
            active_labels_ner = torch.where(active_loss, target_ner.view(-1), torch.tensor(criterion.ignore_index).type_as(target_ner))
            loss_pos = criterion(active_logits_pos, active_labels_pos)
            loss_ner = criterion(active_logits_ner, active_labels_ner)
            loss = (loss_pos + loss_ner) / 2

        return pos_logits, ner_logits, loss