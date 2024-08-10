# Bangla-NER

# NER and POS Tagging API

This project provides a FastAPI application for Named Entity Recognition (NER) and Part-of-Speech (POS) tagging using a pre-trained model based on XLM-RoBERTa.

## Project Structure

- `app/`: Contains the FastAPI application and related files.
  - `main.py`: Entry point for the FastAPI application.
  - `api/routes.py`: Defines the API routes and logic.
  - `models/ner_pos_model.py`: Loads the trained NER and POS tagging model.
  - `models-weight/`: Directory containing the model weights.
  - `utils/predict.py`: Contains functions for making predictions using the model.
- `model/`: Contains the machine learning model code.

  - `config.py`: Configuration settings for the model and tokenizer.
  - `dataset.py`: Defines the `CustomDataset` class for data handling.
  - `model.py`: Defines the `NERPOSModel` class.
  - `train.py`: Contains the training and validation logic.
  - `utils.py`: Utility functions like `seed_everything` and `parse_dataset`.

- `data/`: Directory containing the dataset.

  - `data.tsv`: Dataset file.

- `scripts/`: Contains scripts for training and running the application.

  - `train_model.py`: Script to train the model.
  - `run_app.sh`: Shell script to start the FastAPI server.

- `requirements.txt`: Lists Python dependencies.
- `README.md`: Project documentation.

## Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   Create and activate a virtual environment:
   ```
```
bash
python -m venv venv
source venv/bin/activate # On Windows, use `venv\Scripts\activate`
Install dependencies:
```
```
bash
pip install -r requirements.txt
```

## Usage

### Train the Model:

#### To train the model, run the following script:

```
bash
python scripts/train_model.py
```

#### Start the FastAPI Server:

Use the provided shell script to start the server:

```
bash
./scripts/run_app.sh
```

Alternatively, you can start the server using Uvicorn:

```
bash
uvicorn app.main:app --reload
```

#### Configuration
Configuration settings are defined in model/config.py. You can adjust settings such as maximum sequence length, batch sizes, and model paths here.

#### Model Weights
Ensure that the model weights file (ner_pos_model_xlm_roberta.pt) is located in the app/models-weight/ directory.

#### License
This project is licensed under the MIT License. See the LICENSE file for details.

#### Acknowledgements
XLM-RoBERTa: A transformer-based model for multilingual text.
FastAPI: A modern, fast web framework for building APIs with Python.
