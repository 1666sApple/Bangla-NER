import torch
from torch.utils.data import Dataset
from model.config import Config

class CustomDataset(Dataset):
    def __init__(self, texts, pos_tags, ner_tags):
        self.texts = texts
        self.pos_tags = pos_tags
        self.ner_tags = ner_tags
  
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        texts = self.texts[index]
        pos_tags = self.pos_tags[index]
        ner_tags = self.ner_tags[index]
    
        ids, target_pos, target_ner = [], [], []

        for i, s in enumerate(texts):
            inputs = Config.TOKENIZER.encode(s, add_special_tokens=False)
            input_len = len(inputs)
            ids.extend(inputs)
            target_pos.extend(input_len * [pos_tags[i]])
            target_ner.extend(input_len * [ner_tags[i]])
        
        ids = ids[:Config.MAX_LEN - 2]
        target_pos = target_pos[:Config.MAX_LEN - 2]
        target_ner = target_ner[:Config.MAX_LEN - 2]

        ids = Config.CLS + ids + Config.SEP
        target_pos = Config.VALUE_TOKEN + target_pos + Config.VALUE_TOKEN
        target_ner = Config.VALUE_TOKEN + target_ner + Config.VALUE_TOKEN

        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        padding_len = Config.MAX_LEN - len(ids)
        ids = ids + ([1] * padding_len)
        target_pos = target_pos + ([1] * padding_len)
        target_ner = target_ner + ([1] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target_pos": torch.tensor(target_pos, dtype=torch.long),
            "target_ner": torch.tensor(target_ner, dtype=torch.long)
        }