import sys
import os
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import Config
from model.dataset import CustomDataset
from model.model import NERPOSModel
from model.train import train_fn, val_fn
from model.utils import seed_everything, parse_dataset, get_hyperparameters, create_token_to_id_mapping, create_id_to_token_mapping


def main():
    seed_everything()

    # Load and prepare data
    file_path = 'data/data.tsv'
    df, pos_mapping, ner_mapping, pos_labels, ner_labels = parse_dataset(file_path)

    texts = df['words'].values
    pos_tags = df['pos_tag_id'].values
    ner_tags = df['ner_tag_id'].values
    token_to_id = create_token_to_id_mapping(texts)
    id_to_token = create_id_to_token_mapping(token_to_id)

    train_texts, val_texts, train_pos, val_pos, train_ner, val_ner = train_test_split(
        texts, pos_tags, ner_tags, test_size=0.2, random_state=42
    )

    # Create the training and validation datasets
    train_dataset = CustomDataset(
        texts=train_texts,
        pos_tags=train_pos,
        ner_tags=train_ner,
        token_to_id=token_to_id, 
        max_len=Config.MAX_LEN 
    )

    val_dataset = CustomDataset(
        texts=val_texts,
        pos_tags=val_pos,
        ner_tags=val_ner,
        token_to_id=token_to_id,
        max_len=Config.MAX_LEN
    )

    # Create the DataLoaders for training and validation
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True
    )

    val_data_loader = DataLoader(
        val_dataset,
        batch_size=Config.VAL_BATCH_SIZE,
        shuffle=False
    )

    # If gpu is available, then uncomment the following line
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # If gpu is not available, then use the following line
    # device = torch.device("cpu")
    model = NERPOSModel(
        num_pos=len(pos_mapping),
        num_ner=len(ner_mapping)
    ).to(device)

    optimizer = get_hyperparameters(model)
    total_steps = len(train_data_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    best_f1_ner = 0
    for epoch in range(Config.EPOCHS):
        train_loss = train_fn(train_data_loader, model, optimizer, device, scheduler)
        val_loss, accuracy_pos, accuracy_ner, recall_pos, recall_ner, precision_pos, precision_ner, f1_pos, f1_ner = val_fn(val_data_loader, model, device)

        print(f"Epoch {epoch+1}/{Config.EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"POS - Accuracy: {accuracy_pos:.4f}, Recall: {recall_pos:.4f}, Precision: {precision_pos:.4f}, F1: {f1_pos:.4f}")
        print(f"NER - Accuracy: {accuracy_ner:.4f}, Recall: {recall_ner:.4f}, Precision: {precision_ner:.4f}, F1: {f1_ner:.4f}")

        if f1_ner > best_f1_ner:
            best_f1_ner = f1_ner
            torch.save(model.state_dict(), 'app/models-weight/ner_pos_model.pt')
            print("Model saved!")

    # Save the mappings for inference
    with open('app/models-weight/pos_mapping.pkl', 'wb') as f:
        pickle.dump(pos_mapping, f)
    with open('app/models-weight/ner_mapping.pkl', 'wb') as f:
        pickle.dump(ner_mapping, f)
    
    # Save the token-to-ID mappings
    with open('app/models-weight/token_to_id.pkl', 'wb') as f:
        pickle.dump(token_to_id, f)

    print("Training complete.")

if __name__ == "__main__":
    main()
