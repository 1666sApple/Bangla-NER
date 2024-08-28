import torch
from app.models.ner_pos_model import MODEL, TOKEN_TO_ID, ID_TO_TOKEN, POS_MAPPING, NER_MAPPING, DEVICE

# Define dictionaries for full tag names
POS_FULL_NAMES = {
    'NNC': 'Common Noun',
    'NNP': 'Proper Noun',
    'ADJ': 'Adjective',
    'VF': 'Finite Verb',
    'QF': 'Quantifier',
    'PP': 'Preposition/Particle',
    'VNF': 'Non-Finite Verb',
    'ADV': 'Adverb',
    'PRO': 'Pronoun',
    'CONJ': 'Conjunction',
    'PUNCT': 'Punctuation',
    'DET': 'Determiner',
    'PART': 'Particle',
    'OTH': 'Other',
    'INTJ': 'Interjection'
}

NER_FULL_NAMES = {
    'B-OTH': 'Beginning of Other entity',
    'I-OTH': 'Inside Other entity',
    'B-PER': 'Beginning of Person entity',
    'I-PER': 'Inside Person entity',
    'B-ORG': 'Beginning of Organization entity',
    'I-ORG': 'Inside Organization entity',
    'B-NUM': 'Beginning of Number entity',
    'I-NUM': 'Inside Number entity',
    'B-GPE': 'Beginning of Geopolitical Entity',
    'I-GPE': 'Inside Geopolitical Entity',
    'B-D&T': 'Beginning of Date & Time entity',
    'I-D&T': 'Inside Date & Time entity',
    'B-EVENT': 'Beginning of Event entity',
    'I-EVENT': 'Inside Event entity',
    'B-LOC': 'Beginning of Location entity',
    'I-LOC': 'Inside Location entity',
    'B-UNIT': 'Beginning of Unit entity',
    'I-UNIT': 'Inside Unit entity',
    'B-MISC': 'Beginning of Miscellaneous entity',
    'I-MISC': 'Inside Miscellaneous entity',
    'B-T&T': 'Beginning of Title & Time entity',
    'I-T&T': 'Inside Title & Time entity'
}

def predict_sentence(sentence):
    # Tokenize sentence using the custom token_to_id mapping
    tokens = sentence.split()
    input_ids = [TOKEN_TO_ID.get(token, TOKEN_TO_ID['[PAD]']) for token in tokens]
    input_ids = [TOKEN_TO_ID['[CLS]']] + input_ids + [TOKEN_TO_ID['[SEP]']]
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(DEVICE)

    attention_mask = torch.ones_like(input_ids).to(DEVICE)
    token_type_ids = torch.zeros_like(input_ids).to(DEVICE)

    with torch.no_grad():
        pos_logits, ner_logits, _ = MODEL(input_ids, attention_mask, token_type_ids, None, None)

    pos_preds = torch.argmax(pos_logits, dim=2).cpu().numpy()[0]
    ner_preds = torch.argmax(ner_logits, dim=2).cpu().numpy()[0]

    pos_preds = pos_preds[1:-1]  # Skip [CLS] and [SEP]
    ner_preds = ner_preds[1:-1]  # Skip [CLS] and [SEP]

    result = []
    current_token = ""
    current_pos = pos_preds[0]
    current_ner = ner_preds[0]

    for token, pos_pred, ner_pred in zip(tokens, pos_preds, ner_preds):
        if pos_pred == current_pos and ner_pred == current_ner:
            current_token += token
        else:
            if current_token:
                pos_tag = POS_FULL_NAMES.get(list(POS_MAPPING.keys())[list(POS_MAPPING.values()).index(current_pos)], 'Unknown')
                ner_tag = NER_FULL_NAMES.get(list(NER_MAPPING.keys())[list(NER_MAPPING.values()).index(current_ner)], 'Unknown')
                result.append((current_token, pos_tag, ner_tag))
            current_token = token
            current_pos = pos_pred
            current_ner = ner_pred

    # Add the last token
    if current_token:
        pos_tag = POS_FULL_NAMES.get(list(POS_MAPPING.keys())[list(POS_MAPPING.values()).index(current_pos)], 'Unknown')
        ner_tag = NER_FULL_NAMES.get(list(NER_MAPPING.keys())[list(NER_MAPPING.values()).index(current_ner)], 'Unknown')
        result.append((current_token, pos_tag, ner_tag))

    return result
