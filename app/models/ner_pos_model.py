import torch
from model.model import NERPOSModel
import pickle

def load_model_and_mappings():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load POS and NER mappings
    with open('app/models-weight/pos_mapping.pkl', 'rb') as f:
        pos_mapping = pickle.load(f)
    with open('app/models-weight/ner_mapping.pkl', 'rb') as f:
        ner_mapping = pickle.load(f)

    # Load token mappings
    with open('app/models-weight/token_to_id.pkl', 'rb') as f:
        token_to_id = pickle.load(f)
    id_to_token = {v: k for k, v in token_to_id.items()}

    # Initialize model
    model = NERPOSModel(num_pos=len(pos_mapping), num_ner=len(ner_mapping))
    model.load_state_dict(torch.load('app/models-weight/ner_pos_model.pt', map_location=device))
    model.to(device)
    model.eval()

    return model, token_to_id, id_to_token, pos_mapping, ner_mapping, device

MODEL, TOKEN_TO_ID, ID_TO_TOKEN, POS_MAPPING, NER_MAPPING, DEVICE = load_model_and_mappings()
