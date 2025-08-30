# File Analysis Service

This project is a file analysis service built with Python (FastAPI) and SQLAlchemy. It provides a simple web interface to upload files and view analysis results. The service uses a rule-based approach for initial analysis and includes a script to train a state-of-the-art Transformer model for more advanced sensitive data detection.

## Features

-   **Asynchronous File Analysis:** Upload various file types (text, CSV, JSON, XML, Word, Excel, PDF, images, and archives like .zip and .tar) and receive a task ID immediately. The analysis is performed in the background, and the results can be retrieved without blocking the user.
-   **Automated Remediation:** After detecting sensitive data, the application provides options to automatically redact the information from the file or move the file to a secure quarantine location. High-risk files are automatically quarantined.
-   **Expanded File Support:** The application can now analyze archive files (like `.zip` and `.tar`), which can often contain nested sensitive information.
-   **Real-time Text Analysis:** Paste text directly into a textarea for immediate analysis and feedback.
-   **Enhanced Rule-Based Scanning:** The analysis now includes more sophisticated patterns for common sensitive data (e.g., emails, credit cards, API keys, SSNs, private keys) with weighted risk scoring, as well as entropy-based detection for secrets.
-   **ML-Powered Risk Analysis:** Leverages machine learning models to provide a more nuanced risk assessment. This includes a classification model to predict a risk level (Low, Medium, High) and an anomaly detection model to flag unusual data.
-   **Advanced Rule Management UI:** A sophisticated web interface for managing custom analysis rules, including features for rule testing, versioning, and import/export capabilities.
-   **Customizable Analysis Rules:** Users can define and manage their own regex-based rules for sensitive data detection through the web interface.
-   **Analysis Dashboard:** Provides a visual overview of analysis results, including total files analyzed, risk distribution, and risk by type.
-   **Enhanced Analysis Dashboard:** Provides a more detailed visual overview of analysis results, including a time-series chart of files analyzed over the last 14 days, a table of recent high-risk files, and charts for risk distribution and findings by type.
-   **Machine Learning Ready:** Includes a script (`train_ner_model.py`) to fine-tune a `distilbert-base-uncased` model for Named Entity Recognition (NER) to detect custom sensitive data types.
-   **Image Content Analysis:** Extends beyond OCR to use computer vision models to detect sensitive objects in images, such as credit cards or ID cards.
-   **Improved NER Performance:** Optimized Named Entity Recognition (NER) model inference to prevent hangs and improve stability within the Celery worker, especially for long-running tasks.
-   **Database Integration:** Analysis results are stored in a MongoDB database, providing a scalable and flexible NoSQL solution. Custom rules are stored in a SQLite database using SQLAlchemy.
-   **Modern UI:** The user interface is built with Bootstrap and uses asynchronous JavaScript to poll for results.
-   **Automated Data Source Scanning:** Connect to and scan data from various sources, including:
    -   **Cloud Storage:** Amazon S3, Google Drive, Dropbox.
    -   **Databases:** PostgreSQL, MySQL.
    -   **Version Control Systems:** GitHub, GitLab, to detect secrets in code repositories.
    -   **Dockerfile Scanning:** Scan Dockerfiles for vulnerabilities and hardcoded API keys using Trivy.

## Technologies Used

-   **Backend:** Python 3.9, FastAPI
-   **Task Queue:** Celery, Redis
-   **Database:** MongoDB, SQLite (with SQLAlchemy for ORM)
-   **Frontend:** HTML, Bootstrap, JavaScript
-   **ML/NLP:** PyTorch, Hugging Face Transformers, Scikit-learn
-   **Libraries:** `python-multipart`, `scikit-learn`, `pandas`, `joblib`, `python-docx`, `openpyxl`, `pytesseract`, `Pillow`, `pdfminer.six`, `SQLAlchemy`, `boto3`, `google-api-python-client`, `google-auth-httplib2`, `dropbox`, `psycopg2-binary`, `mysql-connector-python`, `PyGithub`, `python-gitlab`

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

To run the application locally, you will need to start the FastAPI server and a Redis instance. The Celery worker is now integrated into the FastAPI application.

**1. Start Redis:**
```bash
redis-server
```

**2. Start the FastAPI Application (includes Celery worker):**
```bash
PYTHONPATH=. uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Option 2: Run with Docker Compose (Recommended)

For a containerized setup, use Docker Compose. This will build the Docker image and run the application (including the Celery worker), Redis, and MongoDB in separate containers.

```bash
docker-compose up --build
```

This command will build the Docker image (if not already built) and start all the services. The application will be accessible on port `8000`.

### 4. Access the Application

Once the server is running (either locally or via Docker Compose), you can access the application in your web browser at `http://localhost:8000`.

---

## Advanced: Training Custom Models

### NER Model

The application includes a script to fine-tune a Transformer model (`distilbert-base-uncased`) to recognize custom types of sensitive data, including people, organizations, and locations. This is a powerful upgrade from the default rule-based scanner.

1.  **Prepare Your Data:** Ensure your labeled dataset is in `ner_data.jsonl` in the format of `{"tokens": [...], "ner_tags": [...]}`.
2.  **Run the Training Script:** `python train_ner_model_v2.py`
3.  **Integrate the Model:** After training, your model will be saved in the `./ner_model_v2/` directory, where the application will load it.

### Risk Scoring & Anomaly Detection Models

The platform also uses a classification model to predict a risk score and an anomaly detection model to find unusual data.

1.  **Prepare Your Data:** The training data is in `risk_data.csv`. You can add more samples to this file to improve model performance. The features used are `pii_count`, `custom_rules_matches`, and `high_entropy_strings_count`.
2.  **Run the Training Script:**
    ```bash
    python train_risk_models.py
    ```
3.  **Integrate the Models:** The script saves the models as `risk_classifier.joblib` and `anomaly_detector.joblib`. The application automatically loads these files if they exist in the root directory.

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
├── train_ner_model_v2.py # Script for training the NER model
├── train_risk_models.py  # Script for training classification/anomaly models
├── requirements.txt      # Project dependencies
├── docker-compose.yml    # Docker Compose configuration
├── Dockerfile            # Dockerfile for the application
└── README.md             # This file
```

## API Endpoints

-   `POST /uploadfile/`: Upload a file for analysis.
-   `POST /analyze-text/`: Submit text for analysis.
-   `GET /results/{task_id}`: Retrieve the analysis results. The result object will contain the `overall_risk_score` as well as the ML-driven fields `predicted_risk_level` (0=Low, 1=Medium, 2=High) and `is_anomaly` (true/false).
-   `POST /remediate/{task_id}`: Perform remediation actions (redact or quarantine) on a file.
-   ... (and other existing endpoints)

## Known Issues and Future Improvements

-   **NER Model Performance with Large Inputs:** While the NER model hang issue has been mitigated with a timeout, very large text inputs can still lead to timeouts or significant processing delays. Further optimization of the NER pipeline for handling extremely long documents, potentially through more advanced chunking strategies or model quantization, is a future improvement.
-   **Resource Management for ML Models:** The current solution explicitly moves the NER model to CPU. For deployments with GPU resources, optimizing the Docker setup and model loading to leverage GPUs for faster inference would be beneficial.