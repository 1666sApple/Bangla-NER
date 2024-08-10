import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from model.config import Config
from model.dataset import CustomDataset
from model.model import NERPOSModel
from model.train import train_fn, val_fn
from model.utils import seed_everything, parse_dataset, get_hyperparameters
import pickle

def main():
    seed_everything()

    # Load and prepare data
    file_path = 'data/data.tsv'
    df, pos_mapping, ner_mapping, pos_labels, ner_labels = parse_dataset(file_path)

    texts = df['words'].values
    pos_tags = df['pos_tag_id'].values
    ner_tags = df['ner_tag_id'].values

    train_texts, val_texts, train_pos, val_pos, train_ner, val_ner = train_test_split(
        texts, pos_tags, ner_tags, test_size=0.2, random_state=42
    )

    train_dataset = CustomDataset(train_texts, train_pos, train_ner)
    val_dataset = CustomDataset(val_texts, val_pos, val_ner)

    train_data_loader = DataLoader(train_dataset, batch_size=Config.TRAIN_BATCH_SIZE, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=Config.VAL_BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NERPOSModel(num_pos=len(pos_mapping), num_ner=len(ner_mapping)).to(device)

    optimizer = get_hyperparameters(model)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(train_data_loader) * Config.EPOCHS
    )

    # Training loop
    for epoch in range(Config.EPOCHS):
        print(f"Epoch {epoch + 1}/{Config.EPOCHS}")
        train_loss = train_fn(train_data_loader, model, optimizer, device, scheduler)
        val_loss, val_acc_pos, val_acc_ner, val_rec_pos, val_rec_ner, val_prec_pos, val_prec_ner, val_f1_pos, val_f1_ner = val_fn(val_data_loader, model, device)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation POS Accuracy: {val_acc_pos:.4f}, NER Accuracy: {val_acc_ner:.4f}")
        print(f"Validation POS Recall: {val_rec_pos:.4f}, NER Recall: {val_rec_ner:.4f}")
        print(f"Validation POS Precision: {val_prec_pos:.4f}, NER Precision: {val_prec_ner:.4f}")
        print(f"Validation POS F1 Score: {val_f1_pos:.4f}, NER F1 Score: {val_f1_ner:.4f}")

    # Save the model and mappings
    torch.save(model.state_dict(), 'app/models-weight/ner_pos_model.pt')
    with open('app/models-weight/pos_mapping.pkl', 'wb') as f:
        pickle.dump(pos_mapping, f)
    with open('app/models-weight/ner_mapping.pkl', 'wb') as f:
        pickle.dump(ner_mapping, f)

if __name__ == "__main__":
    main()