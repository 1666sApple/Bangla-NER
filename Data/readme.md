# Dataset Details

The dataset contains Named Entity and Parts of Speech tags of about 7000 randomly selected Bangla sentences. The tags are automatically genertaed using a inhouse pre-trained model.

## The dataset is formated in the following format:

```bash
sentence_1
token_1	pos_tag	ner_tag
token_2	pos_tag	ner_tag
token_3	pos_tag	ner_tag

sentence_2
token_1	pos_tag	ner_tag
token_2	pos_tag	ner_tag
token_3	pos_tag	ner_tag
```

## Sample from the dataset:

```bash
#### Sentence 1
Original: `শনিবার (২৭ আগস্ট) রাতে পটুয়াখালী সদর থানার ভারপ্রাপ্ত কর্মকর্তা (ওসি) মো. মনিরুজ্জামান এ তথ্য নিশ্চিত করেছেন।`

| Token         | POS Tag | NER Tag  | Description                        |
|---------------|---------|----------|------------------------------------|
| শনিবার        | NNP     | B-D&T    | Proper Noun, Beginning of Date & Time |
| (২৭           | PUNCT   | B-OTH    | Punctuation, Beginning of Other    |
| আগস্ট)        | NNP     | B-D&T    | Proper Noun, Beginning of Date & Time |
| রাতে          | NNC     | B-D&T    | Common Noun, Beginning of Date & Time |
| পটুয়াখালী     | NNP     | B-GPE    | Proper Noun, Beginning of Geopolitical Entity |
| সদর           | NNC     | I-GPE    | Common Noun, Inside Geopolitical Entity |
| থানার         | NNC     | I-GPE    | Common Noun, Inside Geopolitical Entity |
| ভারপ্রাপ্ত      | ADJ     | B-PER    | Adjective, Beginning of Person     |
| কর্মকর্তা     | NNC     | I-PER    | Common Noun, Inside Person         |
| (ওসি)          | PUNCT   | B-OTH    | Punctuation, Beginning of Other    |
| মো.           | NNP     | B-PER    | Proper Noun, Beginning of Person   |
| মনিরুজ্জামান    | NNP     | I-PER    | Proper Noun, Inside Person         |
| এ              | DET     | B-OTH    | Determiner, Beginning of Other     |
| তথ্য           | NNC     | B-OTH    | Common Noun, Beginning of Other    |
| নিশ্চিত        | ADJ     | B-OTH    | Adjective, Beginning of Other      |
| করেছেন।        | VF      | B-OTH    | Finite Verb, Beginning of Other    |

#### Sentence 2
Original: `বায়ুদূষণ ও স্মার্ট ফোন ছেলেমেয়ে উভয়ের প্রজনন ক্ষমতা হ্রাস করে দিচ্ছে।`

| Token         | POS Tag | NER Tag  | Description                        |
|---------------|---------|----------|------------------------------------|
| বায়ুদূষণ       | NNC     | B-OTH    | Common Noun, Beginning of Other    |
| ও              | CONJ    | B-OTH    | Conjunction, Beginning of Other    |
| স্মার্ট        | NNC     | B-OTH    | Common Noun, Beginning of Other    |
| ফোন           | NNC     | B-OTH    | Common Noun, Beginning of Other    |
| ছেলেমেয়ে      | NNC     | B-PER    | Common Noun, Beginning of Person   |
| উভয়ের         | PRO     | B-OTH    | Pronoun, Beginning of Other        |
| প্রজনন         | NNC     | B-OTH    | Common Noun, Beginning of Other    |
| ক্ষমতা         | NNC     | B-OTH    | Common Noun, Beginning of Other    |
| হ্রাস         | NNC     | B-OTH    | Common Noun, Beginning of Other    |
| করে           | VNF     | B-OTH    | Non-Finite Verb, Beginning of Other|
| দিচ্ছে।       | VF      | B-OTH    | Finite Verb, Beginning of Other    |
```

### POS Tags

- **NNC**: Common Noun
- **NNP**: Proper Noun
- **ADJ**: Adjective
- **VF**: Finite Verb
- **QF**: Quantifier
- **PP**: Preposition/Particle
- **VNF**: Non-Finite Verb
- **ADV**: Adverb
- **PRO**: Pronoun
- **CONJ**: Conjunction
- **PUNCT**: Punctuation
- **DET**: Determiner
- **PART**: Particle
- **OTH**: Other
- **INTJ**: Interjection

### NER Tags

- **B-OTH**: Beginning of Other entity
- **I-OTH**: Inside Other entity
- **B-PER**: Beginning of Person entity
- **I-PER**: Inside Person entity
- **B-ORG**: Beginning of Organization entity
- **I-ORG**: Inside Organization entity
- **B-NUM**: Beginning of Number entity
- **I-NUM**: Inside Number entity
- **B-GPE**: Beginning of Geopolitical Entity
- **I-GPE**: Inside Geopolitical Entity
- **B-D&T**: Beginning of Date & Time entity
- **I-D&T**: Inside Date & Time entity
- **B-EVENT**: Beginning of Event entity
- **I-EVENT**: Inside Event entity
- **B-LOC**: Beginning of Location entity
- **I-LOC**: Inside Location entity
- **B-UNIT**: Beginning of Unit entity
- **I-UNIT**: Inside Unit entity
- **B-MISC**: Beginning of Miscellaneous entity
- **I-MISC**: Inside Miscellaneous entity
- **B-T&T**: Beginning of Title & Time entity
- **I-T&T**: Inside Title & Time entity
