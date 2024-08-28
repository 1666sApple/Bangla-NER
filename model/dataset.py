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
        text = self.texts[index]
        pos_tags = self.pos_tags[index]
        ner_tags = self.ner_tags[index]

        if isinstance(text, list):
            text = ' '.join(text)

        encoding = Config.TOKENIZER.encode_plus(
            text,
            add_special_tokens=True,
            max_length=Config.MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True,
            return_token_type_ids=True,
            return_offsets_mapping=True
        )

        input_ids = encoding['input_ids'].squeeze().tolist()
        attention_mask = encoding['attention_mask'].squeeze().tolist()
        token_type_ids = encoding['token_type_ids'].squeeze().tolist()
        offsets = encoding['offset_mapping'].squeeze().tolist()

        token_to_tag_pos = [Config.VALUE_TOKEN] * len(input_ids)
        token_to_tag_ner = [Config.VALUE_TOKEN] * len(input_ids)
        
        tag_map_pos = [Config.VALUE_TOKEN] * len(input_ids)
        tag_map_ner = [Config.VALUE_TOKEN] * len(input_ids)
        
        for i, (start, end) in enumerate(offsets):
            if start != 0 and end != 0:
                token_index = sum(1 for s, e in offsets[:i] if s != 0 and e != 0) - 1
                if token_index < len(pos_tags):
                    tag_map_pos[i] = pos_tags[token_index]
                    tag_map_ner[i] = ner_tags[token_index]
        
        ids = [Config.CLS] + input_ids[1:-1] + [Config.SEP]
        target_pos = [Config.VALUE_TOKEN] + tag_map_pos[1:-1] + [Config.VALUE_TOKEN]
        target_ner = [Config.VALUE_TOKEN] + tag_map_ner[1:-1] + [Config.VALUE_TOKEN]
        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        padding_len = Config.MAX_LEN - len(ids)
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
