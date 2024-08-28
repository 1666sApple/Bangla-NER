import random
import numpy as np
import torch
import pandas as pd
from model.config import Config

def seed_everything(seed=64):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_dataset(file_path):
    sentences, words, pos_tags, ner_tags = [], [], [], []
    with open(file_path, 'r', encoding='utf-8') as file:
        current_sentence, current_pos, current_ner = [], [], []
        for line in file:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 3:
                    token, pos, ner = parts
                    current_sentence.append(token)
                    current_pos.append(pos)
                    current_ner.append(ner)
            else:
                if current_sentence:
                    sentences.append(" ".join(current_sentence))
                    words.append(current_sentence)
                    pos_tags.append(current_pos)
                    ner_tags.append(current_ner)
                    current_sentence, current_pos, current_ner = [], [], []
        if current_sentence:
            sentences.append(" ".join(current_sentence))
            words.append(current_sentence)
            pos_tags.append(current_pos)
            ner_tags.append(current_ner)
    df = pd.DataFrame({'words': words, 'pos': pos_tags, 'ner': ner_tags})
    pos_set = set(tag for sublist in df['pos'] for tag in sublist)
    ner_set = set(tag for sublist in df['ner'] for tag in sublist)
    
    pos_labels = sorted(pos_set)
    ner_labels = sorted(ner_set)
    
    pos_mapping = {tag: idx for idx, tag in enumerate(pos_labels)}
    ner_mapping = {tag: idx for idx, tag in enumerate(ner_labels)}
    
    df['pos_tag_id'] = df['pos'].apply(lambda tags: [pos_mapping[tag] for tag in tags])
    df['ner_tag_id'] = df['ner'].apply(lambda tags: [ner_mapping[tag] for tag in tags])
    
    return df, pos_mapping, ner_mapping, pos_labels, ner_labels

def get_hyperparameters(model, ff=True, weight_decay=0.01):
    if ff:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": weight_decay},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=5e-5)
        return optimizer

def create_token_to_id_mapping(texts):
    token_set = set()
    for text in texts:
        if isinstance(text, list):
            tokens = text
        else:
            tokens = text.split()
        token_set.update(tokens)
    
    token_to_id = {token: idx + 3 for idx, token in enumerate(token_set)}
    token_to_id['[CLS]'] = Config.CLS
    token_to_id['[SEP]'] = Config.SEP
    token_to_id['[PAD]'] = Config.VALUE_TOKEN
    return token_to_id

def create_id_to_token_mapping(token_to_id):
    return {v: k for k, v in token_to_id.items()}