import torch
from torch.utils.data import Dataset
from model.config import Config

class CustomDataset(Dataset):
    def __init__(self, texts, pos_tags, ner_tags, token_to_id, max_len):
        self.texts = texts
        self.pos_tags = pos_tags
        self.ner_tags = ner_tags
        self.token_to_id = token_to_id
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        pos_tags = self.pos_tags[index]
        ner_tags = self.ner_tags[index]

        if isinstance(text, list):
            tokens = text
        else:
            tokens = text.split()

        input_ids = [self.token_to_id.get(token, Config.VALUE_TOKEN) for token in tokens]

        tag_map_pos = pos_tags[:len(input_ids)]
        tag_map_ner = ner_tags[:len(input_ids)]

        ids = [Config.CLS] + input_ids + [Config.SEP]
        target_pos = [Config.VALUE_TOKEN] + tag_map_pos + [Config.VALUE_TOKEN]
        target_ner = [Config.VALUE_TOKEN] + tag_map_ner + [Config.VALUE_TOKEN]
        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        padding_len = self.max_len - len(ids)
        ids.extend([Config.VALUE_TOKEN] * padding_len)
        target_pos.extend([Config.VALUE_TOKEN] * padding_len)
        target_ner.extend([Config.VALUE_TOKEN] * padding_len)
        mask.extend([0] * padding_len)
        token_type_ids.extend([0] * padding_len)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target_pos": torch.tensor(target_pos, dtype=torch.long),
            "target_ner": torch.tensor(target_ner, dtype=torch.long)
        }
