import torch
from app.models.ner_pos_model import MODEL, TOKEN_TO_ID, ID_TO_TOKEN, POS_MAPPING, NER_MAPPING, DEVICE

# Define dictionaries for full tag names
POS_FULL_NAMES = {
    'NNC': 'Common Noun (NNC)',
    'NNP': 'Proper Noun (NNP)',
    'ADJ': 'Adjective (ADJ)',
    'VF': 'Finite Verb (VF)',
    'QF': 'Quantifier (QF)',
    'PP': 'Preposition/Particle (PP)',
    'VNF': 'Non-Finite Verb (VNF)',
    'ADV': 'Adverb (ADV)',
    'PRO': 'Pronoun (PRO)',
    'CONJ': 'Conjunction (CONJ)',
    'PUNCT': 'Punctuation (PUNCT)',
    'DET': 'Determiner (DET)',
    'PART': 'Particle (PART)',
    'OTH': 'Other (OTH)',
    'INTJ': 'Interjection (INTJ)'
}

NER_FULL_NAMES = {
    'B-OTH': 'Beginning of Other entity (B-OTH)',
    'I-OTH': 'Inside Other entity (I-OTH)',
    'B-PER': 'Beginning of Person entity (B-PER)',
    'I-PER': 'Inside Person entity (I-PER)',
    'B-ORG': 'Beginning of Organization entity (B-ORG)',
    'I-ORG': 'Inside Organization entity (I-ORG)',
    'B-NUM': 'Beginning of Number entity (B-NUM)',
    'I-NUM': 'Inside Number entity (I-NUM)',
    'B-GPE': 'Beginning of Geopolitical Entity (B-GPE)',
    'I-GPE': 'Inside Geopolitical Entity (I-GPE)',
    'B-D&T': 'Beginning of Date & Time entity (B-D&T)',
    'I-D&T': 'Inside Date & Time entity (I-D&T)',
    'B-EVENT': 'Beginning of Event entity (B-EVENT)',
    'I-EVENT': 'Inside Event entity (I-EVENT)',
    'B-LOC': 'Beginning of Location entity (B-LOC)',
    'I-LOC': 'Inside Location entity (I-LOC)',
    'B-UNIT': 'Beginning of Unit entity (B-UNIT)',
    'I-UNIT': 'Inside Unit entity (I-UNIT)',
    'B-MISC': 'Beginning of Miscellaneous entity (B-MISC)',
    'I-MISC': 'Inside Miscellaneous entity (I-MISC)',
    'B-T&T': 'Beginning of Title & Time entity (B-T&T)',
    'I-T&T': 'Inside Title & Time entity (I-T&T)'
}

def predict_sentence(sentence):
    # Tokenize the input sentence and map tokens to IDs
    tokens = sentence.split()
    input_ids = [TOKEN_TO_ID.get(token, TOKEN_TO_ID.get('[UNK]')) for token in tokens]
    
    # Add [CLS] and [SEP] tokens
    input_ids = [TOKEN_TO_ID.get('[CLS]')] + input_ids + [TOKEN_TO_ID.get('[SEP]')]
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(DEVICE)

    # Create attention mask and token type IDs
    attention_mask = torch.ones_like(input_ids).to(DEVICE)
    token_type_ids = torch.zeros_like(input_ids).to(DEVICE)

    # Perform inference
    with torch.no_grad():
        pos_logits, ner_logits, _ = MODEL(input_ids, attention_mask, token_type_ids, None, None)

    # Get predicted POS and NER tags
    pos_preds = torch.argmax(pos_logits, dim=2).cpu().numpy()[0]
    ner_preds = torch.argmax(ner_logits, dim=2).cpu().numpy()[0]

    # Convert input IDs back to tokens
    tokens = [ID_TO_TOKEN.get(i, '[UNK]') for i in input_ids[0].cpu().numpy()]

    # Exclude [CLS] and [SEP] tokens and their predictions
    tokens = tokens[1:-1]
    pos_preds = pos_preds[1:-1]
    ner_preds = ner_preds[1:-1]

    # Convert predicted IDs to POS and NER tags using the mappings
    pos_tags = [list(POS_MAPPING.keys())[list(POS_MAPPING.values()).index(tag)] for tag in pos_preds]
    ner_tags = [list(NER_MAPPING.keys())[list(NER_MAPPING.values()).index(tag)] for tag in ner_preds]

    # Map the tags to their full names
    result = []
    for token, pos_tag, ner_tag in zip(tokens, pos_tags, ner_tags):
        pos_full_name = POS_FULL_NAMES.get(pos_tag, 'Unknown')
        ner_full_name = NER_FULL_NAMES.get(ner_tag, 'Unknown')
        result.append((token, pos_full_name, ner_full_name))

    return result
