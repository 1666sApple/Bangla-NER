from transformers import AutoTokenizer

class Config:
    CLS = 0  # XLM-RoBERTa CLS token
    SEP = 2  # XLM-RoBERTa SEP token
    VALUE_TOKEN = 1  # XLM-RoBERTa padding token
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 4
    VAL_BATCH_SIZE = 2
    EPOCHS = 9
    TOKENIZER = AutoTokenizer.from_pretrained("xlm-roberta-base")