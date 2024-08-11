# Bangla-NER-POS

This project provides a FastAPI application for Named Entity Recognition (NER) and Part-of-Speech (POS) tagging using a pre-trained model based on XLM-RoBERTa. The application allows users to input Bangla text and receive annotated outputs with corresponding NER and POS tags.

## Project Structure

The project is organized into the following directories and files:

```bash
Bangla-NER-POS/
│
├── app/            # Entry point for FastAPI application
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py         # Defines the API routes and logic
│   ├── models/
│   │   ├── __init__.py
│   │   └── ner_pos_model.py  # Load the trained model for inference
│   ├── models-weight/
│   │   ├── ner_pos_model.pt
│   │   ├── ner_mapping.pkl
│   │   ├── ner_mapping.json
│   │   ├── pos_mapping.pkl
│   │   └── pos_mapping.json
│   │
│   ├── static/
│   │   ├── script.js
│   │   └── style.css
│   ├── templates/
│   │   ├── index.html
│   │   └── predict.html
│   ├── utils/
│   │   ├── __init__.py
│   │   └── predict.py
│   ├── __init__.py
│   └── main.py           # Function for making predictions using the model
│
├── model/
│   ├── __init__.py
│   ├── config.py             # Configuration settings
│   ├── dataset.py            # CustomDataset class
│   ├── model.py              # NERPOSModel class
│   ├── train.py              # Training and validation logic
│   └── utils.py              # Utility functions (e.g., seed_everything, parse_dataset)
│
├── screenshots/
│   ├── app-command.png
│   ├── home.png
│   ├── eval.png
│   └── prediction.png
│
├── scripts/
│   ├── train_model.py        # Script to train the model
│   └── run_app.sh            # Shell script to start the FastAPI server
│
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

- **`app/`**: Contains the FastAPI application and related files.

  - **`main.py`**: Entry point for the FastAPI application.
  - **`api/routes.py`**: Defines the API routes and logic.
  - **`models/ner_pos_model.py`**: Loads the trained NER and POS tagging model.
  - **`models-weight/`**: Directory containing the model weights.
  - **`utils/predict.py`**: Contains functions for making predictions using the model.

- **`model/`**: Contains the machine learning model code.

  - **`config.py`**: Configuration settings for the model and tokenizer.
  - **`dataset.py`**: Defines the `CustomDataset` class for data handling.
  - **`model.py`**: Defines the `NERPOSModel` class.
  - **`train.py`**: Contains the training and validation logic.
  - **`utils.py`**: Utility functions like `seed_everything` and `parse_dataset`.

- **`data/`**: Directory containing the dataset.

  - **`data.tsv`**: Dataset file used for training and validation.

- **`scripts/`**: Contains scripts for training and running the application.

  - **`train_model.py`**: Script to train the model.
  - **`run_app.sh`**: Shell script to start the FastAPI server.

- **`screenshots/`**: Contains screenshots of the application in action.

- **`requirements.txt`**: Lists Python dependencies required for the project.
- **`README.md`**: Project documentation, including setup and usage instructions.

## Installation

### 1. Clone the Repository:

First, clone the repository to your local machine:

```bash
git clone https://github.com/1666sApple/Bangla-NER-POS.git
cd Bangla-NER-POS
```

### 2. Create and Activate a Virtual Environment:

Create a virtual environment to manage dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies:

Install the required Python packages:

```python
pip install -r requirements.txt
```

#### Explanation for for each dependency in `requirements.txt`:

1. **transformers**: Library by Hugging Face for NLP tasks, providing pre-trained models like BERT and GPT, useful for NER and POS tagging.

2. **torch**: Deep learning framework (PyTorch) used for building and training neural network models, acting as the backend for models.

3. **pandas**: Data manipulation and analysis library for handling structured data in dataframes, essential for dataset processing.

4. **scikit-learn**: Machine learning library offering tools for data preprocessing and model evaluation, including utilities like LabelEncoder for encoding labels.

5. **numpy**: Fundamental package for numerical computing in Python, supporting multi-dimensional arrays and mathematical operations.

6. **fastapi**: High-performance web framework for building APIs with Python 3.7+ based on type hints, useful for exposing models as REST APIs.

7. **uvicorn**: ASGI server implementation for serving FastAPI applications, enabling asynchronous web request handling.

8. **gunicorn**: WSGI HTTP server for running Python web applications, commonly used for deploying Flask applications in production.

9. **Jinja2**: Templating engine for Python used by web frameworks like Flask to dynamically render HTML templates.

10. **pickle**: Standard Python library for serializing and deserializing Python objects, useful for saving models and data structures. Typically not listed in requirements.txt as it's part of the standard library.

11. **pydantic**: Data validation and settings management library using Python type annotations, used by FastAPI to validate and manage data structures.

## Usage

### 1. Train the Model:

To train the model using the provided dataset, run the following script:

```python
python3 scripts/train_model.py
```

### 2. Start the FastAPI Server:

To run the FastAPI server, follow these steps:

1. **Grant Execute Permission:**

Make sure the `run_app.sh` script has execute permissions. You can grant these permissions by running

```bash
chmod +x scripts/run_app.sh
```

2. **Verify the Permissions**:

Check that the script has execute permissions by running:

```bash
ls -l scripts/run_app.sh
```

The output should include an `x` in the permission string, indicating execute permissions (e.g., `-rwxr-xr-x`).

3. **Run the Script:**

Run the script to start the FastAPI server :

```bash
./scripts/run_app.sh
```

The server will start on http://0.0.0.0:8000 and will automatically reload on code changes.

Alternatively, you can start the server directly using Uvicorn:

```bash
uvicorn app.main:app --reload
```

## 3. Access the API:

Once the server is running, you can access the API at [http://127.0.0.1:8000](http://127.0.0.1:8000). The API will allow you to submit Bangla text and receive NER and POS tags in the response.

## Configuration

Configuration settings for the model, tokenizer, and training parameters are defined in `model/config.py`. You can adjust settings such as:

- Maximum sequence length (`MAX_LEN`)
- Batch sizes (`TRAIN_BATCH_SIZE`, `VAL_BATCH_SIZE`)
- Number of epochs (`EPOCHS`)
- Tokenizer model (`TOKENIZER`)
- Model weights paths (`MODEL_PATH`)

## Model Weights

Ensure that the model weights file (`ner_pos_model.pt`) is placed in the `app/models-weight/` directory. The application will load these weights to perform NER and POS tagging.

**_ps:_** **Here training the `python3 scripts/train_model.py` script will automatically create `ner_pos_model.pt` in the mentioned directory. However, make sure that the project is structured in the similar manner and strictly follows the naming conventions.**

**_Initially, `ner_pos_model.pt` is not added. You have to train the script at least once to obtain the model._**

## Screenshots

To give users a better understanding of how the application works, you can include screenshots of the API in action. Here are some suggestions:

####  ***1. Starting App:***

A screenshot of cli commands to start the application:

![CLI command](screenshots/app-command.png?raw=true "Screenshot of the CLI command"))

####  ***2. Home Page:***

A screenshot of the landing page of the web application:

![Homepage](screenshots/home.png?raw=true "Screenshot of the Homepage")

####  ***3. NER/POS Tagging Result:***

A screenshot showing the input Bangla text and the corresponding NER and POS tags returned by the API:

![Prediction](screenshots/prediction.png?raw=true "Screenshot of the NER/POS Tagging Result")

####  ***4. Training Output:***

A screenshot of the terminal output during model training, showing the loss and accuracy metrics over epochs.

![Training](screenshots/training.png?raw=true "Screenshot of the Training Output")

####  ***5. Output Evaluation Matrices:***

A screenshot of the terminal output after model training, showing the loss and accuracy metrics over epochs.

![Evaluation](screenshots/eval.png?raw=true "Screenshot of Evaluation Matrices")

## License

This project is not licensed under any institutional licenses.

## Acknowledgements

- **XLM-RoBERTa**: A transformer-based model for multilingual text. [Link](https://huggingface.co/docs/transformers/model_doc/xlm-roberta)
- **FastAPI**: A modern, fast web framework for building APIs with Python. [Link](https://fastapi.tiangolo.com/)

## Contact

For questions or feedback, please contact [1666sApple](https://github.com/1666sApple).

## Contributing

Contributions are welcome! Please follow the standard GitHub workflow: fork the repository, create a new branch, commit your changes, and submit a pull request. Make sure your code adheres to the project's coding standards and includes appropriate tests.
