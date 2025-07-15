# File Analysis Service

This project is a file analysis service built with Python (FastAPI) and SQLAlchemy. It provides a simple web interface to upload files and view analysis results. The analysis model scans for sensitive data patterns, such as email addresses and credit card numbers.

## Features

-   **File Upload and Analysis:** Upload files through a web interface and receive a risk score and detailed findings.
-   **Sensitive Data Detection:** The analysis model scans for common sensitive data patterns using regular expressions.
-   **Database Integration:** Analysis results are stored in a SQLite database using SQLAlchemy, which allows for easy migration to other databases like PostgreSQL.
-   **Modern UI:** The user interface is built with Bootstrap and uses asynchronous JavaScript to provide a smooth user experience.

## Technologies Used

-   **Backend:** Python 3.9, FastAPI
-   **Database:** SQLite (with SQLAlchemy for ORM)
-   **Frontend:** HTML, Bootstrap, JavaScript
-   **Libraries:** `python-multipart`

## Getting Started

Follow these steps to set up and run the application locally.

### Prerequisites

Ensure you have the following installed on your system:

-   Python 3.7+
-   pip (Python package installer)

### 1. Clone the Repository (if applicable)

If you haven't already, clone this project to your local machine:

```bash
git clone <repository-url>
cd risk_control_platform
```

### 2. Install Dependencies

Install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

### 3. Run the Application

Start the FastAPI application using Uvicorn:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

This command will start the server and automatically reload it when you make changes to the code.

### 4. Access the Application

Once the server is running, you can access the application in your web browser at `http://localhost:8000`.

## Usage

1.  **Open the web interface:** Navigate to `http://localhost:8000` in your browser.
2.  **Select a file:** Click the "Select a file to analyze" button and choose a file from your local machine.
3.  **Analyze the file:** Click the "Analyze" button to upload and analyze the file.
4.  **View the results:** The analysis results, including the risk score and any findings, will be displayed on the page.

## Project Structure

```
risk_control_platform/
├── app/
│   ├── __init__.py
│   ├── analysis.py       # Contains the file analysis logic
│   ├── crud.py           # Contains the database operations
│   ├── database.py       # Contains the database connection setup
│   ├── main.py           # The main FastAPI application
│   ├── models.py         # Contains the database models
│   ├── schemas.py        # Contains the Pydantic models
│   └── static/
│       └── index.html    # The main HTML file for the UI
├── requirements.txt      # The project dependencies
└── README.md             # This file
```