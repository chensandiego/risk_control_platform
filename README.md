# File Analysis Service

This project is a file analysis service built with Python (FastAPI) and SQLAlchemy. It provides a simple web interface to upload files and view analysis results. The service uses a rule-based approach for initial analysis and includes a script to train a state-of-the-art Transformer model for more advanced sensitive data detection.

## Features

-   **Asynchronous File Analysis:** Upload various file types (text, CSV, JSON, XML, Word, Excel, PDF, images) and receive a task ID immediately. The analysis is performed in the background, and the results can be retrieved without blocking the user.
-   **Real-time Text Analysis:** Paste text directly into a textarea for immediate analysis and feedback.
-   **Enhanced Rule-Based Scanning:** The analysis now includes more sophisticated patterns for common sensitive data (e.g., emails, credit cards, API keys, SSNs, private keys) with weighted risk scoring, as well as entropy-based detection for secrets.
-   **Customizable Analysis Rules:** Users can define and manage their own regex-based rules for sensitive data detection through the web interface.
-   **Analysis Dashboard:** Provides a visual overview of analysis results, including total files analyzed, risk distribution, and risk by type.
-   **Machine Learning Ready:** Includes a script (`train_ner_model.py`) to fine-tune a `distilbert-base-uncased` model for Named Entity Recognition (NER) to detect custom sensitive data types.
-   **Image Content Analysis:** Extends beyond OCR to use computer vision models to detect sensitive objects in images, such as credit cards or ID cards.
-   **Database Integration:** Analysis results are stored in a MongoDB database, providing a scalable and flexible NoSQL solution. Custom rules are stored in a SQLite database using SQLAlchemy.
-   **Modern UI:** The user interface is built with Bootstrap and uses asynchronous JavaScript to poll for results.

## Technologies Used

-   **Backend:** Python 3.9, FastAPI
-   **Task Queue:** Celery, Redis
-   **Database:** MongoDB, SQLite (with SQLAlchemy for ORM)
-   **Frontend:** HTML, Bootstrap, JavaScript
-   **ML/NLP:** PyTorch, Hugging Face Transformers
-   **Libraries:** `python-multipart`, `scikit-learn`, `python-docx`, `openpyxl`, `pytesseract`, `Pillow`, `pdfminer.six`, `SQLAlchemy`

## Architecture

The application uses a client-server architecture with a background task queue for processing file analyses. This ensures that the application remains responsive, even when analyzing large files.

1.  **Client (Browser):** The user uploads a file or submits text through the web interface.
2.  **FastAPI Server:** The server receives the request and creates a new analysis task.
3.  **Celery Task Queue:** The task is sent to a Celery worker for processing.
4.  **Redis:** Redis serves as the message broker and result backend for Celery.
5.  **MongoDB:** Stores the detailed analysis results.
6.  **SQLite:** Stores custom analysis rules.
7.  **Client Polling:** The client polls the server for the analysis results using the task ID.

## Getting Started

Follow these steps to set up and run the application locally.

### Prerequisites

Ensure you have the following installed on your system:

-   Python 3.7+
-   pip (Python package installer)
-   Docker and Docker Compose (for containerized deployment)
-   **Tesseract OCR Engine:** Required for image and PDF analysis. Install it via your system's package manager (e.g., `brew install tesseract` on macOS, `sudo apt install tesseract-ocr` on Ubuntu/Debian) or download from [Tesseract GitHub page](https://tesseract-ocr.github.io/tessdoc/Downloads.html).

### 1. Clone the Repository (if applicable)

If you haven't already, clone this project to your local machine:

```bash
git clone <repository-url>
cd risk_control_platform
```

### 2. Install Dependencies

Install the required Python packages using `pip`.

```bash
pip install -r requirements.txt
```

### 3. Run the Application

#### Option 1: Run Locally (Python)

To run the application locally, you will need to start the FastAPI server, a Redis instance, and a Celery worker.

**1. Start Redis:**
```bash
redis-server
```

**2. Start the Celery Worker:**
```bash
celery -A celery_app worker --loglevel=info
```

**3. Start the FastAPI Application:**
```bash
PYTHONPATH=. uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Option 2: Run with Docker Compose (Recommended)

For a containerized setup, use Docker Compose. This will build the Docker image and run the application, Redis, MongoDB, and a Celery worker in separate containers.

```bash
docker-compose up --build
```

This command will build the Docker image (if not already built) and start all the services. The application will be accessible on port `8000`.

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
│   ├── crud.py           # Database operations for MongoDB
│   ├── database.py       # Database connection setup (MongoDB and SQLite)
│   ├── dashboard.py      # Logic for generating dashboard data
│   ├── main.py           # Main FastAPI application
│   ├── models.py         # Database models (MongoDB Pydantic and SQLite SQLAlchemy)
│   ├── rules_crud.py     # Database operations for SQLite custom rules
│   ├── schemas.py        # Pydantic models
│   └── static/
│       └── index.html    # Main HTML file for the UI
├── celery_app.py         # Celery application setup
├── train_ner_model.py    # Script for training the NER model
├── requirements.txt      # Project dependencies
├── docker-compose.yml    # Docker Compose configuration
├── Dockerfile            # Dockerfile for the application
└── README.md             # This file
```
