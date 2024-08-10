import torch
from model.model import NERPOSModel
from transformers import AutoTokenizer
import pickle

def load_model_and_mappings():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load POS and NER mappings
    with open('app/models-weight/pos_mapping.pkl', 'rb') as f:
        pos_mapping = pickle.load(f)
    with open('app/models-weight/ner_mapping.pkl', 'rb') as f:
        ner_mapping = pickle.load(f)

    # Initialize model
    model = NERPOSModel(num_pos=len(pos_mapping), num_ner=len(ner_mapping))
    model.load_state_dict(torch.load('app/models-weight/ner_pos_model.pt', map_location=device))
    model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    return model, tokenizer, pos_mapping, ner_mapping, device

MODEL, TOKENIZER, POS_MAPPING, NER_MAPPING, DEVICE = load_model_and_mappings()