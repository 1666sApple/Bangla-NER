import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    total_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        optimizer.zero_grad()
        _, _, loss = model(**data)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        torch.cuda.empty_cache()
    return total_loss / len(data_loader)

def val_fn(data_loader, model, device):
    model.eval()
    losses = []
    all_preds_pos = []
    all_targets_pos = []
    all_preds_ner = []
    all_targets_ner = []

    with torch.no_grad():
        for batch in data_loader:
            ids = batch['ids'].to(device)
            mask = batch['mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            target_pos = batch['target_pos'].to(device)
            target_ner = batch['target_ner'].to(device)
            
            pos, ner, loss = model(ids, mask, token_type_ids, target_pos, target_ner)
            losses.append(loss.item())
            
            all_preds_pos.extend(pos.argmax(-1).detach().cpu().numpy().flatten())
            all_targets_pos.extend(target_pos.detach().cpu().numpy().flatten())
            all_preds_ner.extend(ner.argmax(-1).detach().cpu().numpy().flatten())
            all_targets_ner.extend(target_ner.detach().cpu().numpy().flatten())

    avg_loss = np.mean(losses)
    
    accuracy_pos = accuracy_score(all_targets_pos, all_preds_pos)
    accuracy_ner = accuracy_score(all_targets_ner, all_preds_ner)
    recall_pos = recall_score(all_targets_pos, all_preds_pos, average='weighted')
    recall_ner = recall_score(all_targets_ner, all_preds_ner, average='weighted')
    precision_pos = precision_score(all_targets_pos, all_preds_pos, average='weighted')
    precision_ner = precision_score(all_targets_ner, all_preds_ner, average='weighted')
    f1_pos = f1_score(all_targets_pos, all_preds_pos, average='weighted')
    f1_ner = f1_score(all_targets_ner, all_preds_ner, average='weighted')

    return avg_loss, accuracy_pos, accuracy_ner, recall_pos, recall_ner, precision_pos, precision_ner, f1_pos, f1_ner