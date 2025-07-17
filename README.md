# File Analysis Service

This project is a file analysis service built with Python (FastAPI) and SQLAlchemy. It provides a simple web interface to upload files and view analysis results. The service uses a rule-based approach for initial analysis and includes a script to train a state-of-the-art Transformer model for more advanced sensitive data detection.

## Features

-   **File Upload and Analysis:** Upload files through a web interface and receive a detailed risk score and comprehensive findings based on enhanced analysis patterns.
-   **Enhanced Rule-Based Scanning:** The analysis now includes more sophisticated patterns for common sensitive data (e.g., emails, credit cards, API keys, SSNs, private keys) with weighted risk scoring.
-   **Machine Learning Ready:** Includes a script (`train_ner_model.py`) to fine-tune a `distilbert-base-uncased` model for Named Entity Recognition (NER) to detect custom sensitive data types.
-   **Database Integration:** Analysis results are stored in a SQLite database using SQLAlchemy, allowing for easy migration to other databases like PostgreSQL.
-   **Modern UI:** The user interface is built with Bootstrap and uses asynchronous JavaScript for a smooth user experience.

## Technologies Used

-   **Backend:** Python 3.9, FastAPI
-   **Database:** SQLite (with SQLAlchemy for ORM)
-   **Frontend:** HTML, Bootstrap, JavaScript
-   **ML/NLP:** PyTorch, Hugging Face Transformers
-   **Libraries:** `python-multipart`, `scikit-learn`

## Getting Started

Follow these steps to set up and run the application locally.

### Prerequisites

Ensure you have the following installed on your system:

-   Python 3.7+
-   pip (Python package installer)
-   Docker and Docker Compose (for containerized deployment)

### 1. Clone the Repository (if applicable)

If you haven't already, clone this project to your local machine:

```bash
git clone <repository-url>
cd risk_control_platform
```

### 2. Install Dependencies

Install the required Python packages using `pip`. This includes FastAPI, Uvicorn, and the machine learning libraries.

```bash
pip install -r requirements.txt
```

### 3. Run the Application

#### Option 1: Run Locally (Python)

Start the FastAPI application using Uvicorn:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

This command starts the server. It will use the default rule-based analysis engine.

#### Option 2: Run with Docker Compose (Recommended)

For a containerized setup, use Docker Compose. This will build the Docker image and run the application in a container.

```bash
docker-compose up --build
```

This command will build the Docker image (if not already built) and start the FastAPI application. The application will be accessible on port 8000.

### 4. Access the Application

Once the server is running (either locally or via Docker Compose), you can access the application in your web browser at `http://localhost:8000`.

---

## Advanced: Training a Custom NER Model

The application includes a script to fine-tune a Transformer model (`distilbert-base-uncased`) to recognize custom types of sensitive data. This is a powerful upgrade from the default rule-based scanner.

### How It Works

The `train_ner_model.py` script performs the following steps:
1.  Loads a small, sample labeled dataset (you should replace this with your own data).
2.  Loads the pre-trained `distilbert-base-uncased` model and tokenizer from Hugging Face.
3.  Tokenizes the text and aligns the labels with the model's tokenizer.
4.  Fine-tunes the model on your labeled data using the Hugging Face `Trainer` API.
5.  Saves the resulting fine-tuned model and tokenizer to a local directory (`./ner_model/`).

### How to Train the Model

1.  **Prepare Your Data:** Open `train_ner_model.py` and replace the example `texts` and `labels` with your own labeled dataset. The more high-quality data you provide, the better the model will perform.

2.  **Run the Training Script:** Execute the script from your terminal.

    ```bash
    python train_ner_model.py
    ```
    This process may take some time and is computationally intensive. Using a GPU is recommended for larger datasets.

3.  **Integrate the Model:** After training is complete, your fine-tuned model will be saved in the `./ner_model/` directory. You can then modify `app/analysis.py` to load this model and use it for inference instead of the default regex-based functions.

## Project Structure

```
risk_control_platform/
├── app/
│   ├── analysis.py       # Contains the file analysis logic
│   ├── crud.py           # Database operations
│   ├── database.py       # Database connection setup
│   ├── main.py           # Main FastAPI application
│   ├── models.py         # Database models
│   ├── schemas.py        # Pydantic models
│   └── static/
│       └── index.html    # Main HTML file for the UI
├── train_ner_model.py    # Script for training the NER model
├── requirements.txt      # Project dependencies
├── docker-compose.yml    # Docker Compose configuration
├── Dockerfile            # Dockerfile for the application
└── README.md             # This file
```