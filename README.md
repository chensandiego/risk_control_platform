# Big Data Intelligent Risk Control Platform

This project implements a Big Data Intelligent Risk Control Platform using Python (FastAPI), MongoDB, Redis, and Docker. It provides functionalities for URL anti-fraud, traffic anti-fraud (rate-limiting), and a foundation for machine learning-based risk analysis.

## Features

-   **URL Anti-Fraud:** Checks URLs against a blocklist, performs heuristic analysis to identify suspicious patterns, and leverages the **AbuseIPDB API** to check the URL's IP address against a database of known malicious IPs.
-   **Traffic Anti-Fraud:** Implements rate-limiting to protect API endpoints from excessive requests and logs blocked traffic.
-   **Dynamic Keyword Management:** Suspicious keywords for heuristic analysis are stored and managed in MongoDB, allowing for real-time updates without application redeployment.
-   **Machine Learning Foundation:** Provides an endpoint to log and label URLs with extracted features, building a dataset for future ML model training.
-   **Web Dashboard:** A simple web interface to check URLs and monitor real-time traffic events.

## Technologies Used

-   **Backend:** Python 3.9, FastAPI
-   **Database:** MongoDB
-   **Caching/Real-time Data:** Redis
-   **Containerization:** Docker, Docker Compose
-   **External APIs:** AbuseIPDB
-   **Libraries:** `requests`, `python-dotenv`

## Getting Started

Follow these steps to set up and run the application locally using Docker Compose.

### Prerequisites

Ensure you have the following installed on your system:

-   [Docker Desktop](https://www.docker.com/products/docker-desktop)

### 1. Clone the Repository (if applicable)

If you haven't already, clone this project to your local machine:

```bash
git clone <repository-url>
cd risk_control_platform
```

### 2. Configure AbuseIPDB API Key

Create a `.env` file in the `risk_control_platform` directory:

```bash
touch .env
```

Open the `.env` file and add your AbuseIPDB API key:

```
ABUSEIPDB_API_KEY="YOUR_ABUSEIPDB_API_KEY"
```

Replace `"YOUR_ABUSEIPDB_API_KEY"` with your actual API key.

### 3. Start the Docker Containers

Navigate to the `risk_control_platform` directory (where `docker-compose.yml` is located) and run the following command to build the Docker images and start all services in detached mode:

```bash
docker compose up -d --build
```

This command will:
-   Build the `fastapi-app` Docker image.
-   Start the `mongodb` container.
-   Start the `redis` container.
-   Start the `fastapi-app` container, connecting it to MongoDB and Redis.

### 4. Verify Application Status

After running `docker compose up -d --build`, you can check the status of your containers:

```bash
docker compose ps
```

You should see `Up` status for `fastapi-app`, `mongodb`, and `redis`.

### 5. Access the Application

Once all services are running, you can access the application:

-   **Web Dashboard:** Open your web browser and go to `http://localhost:8000`
-   **API Documentation (Swagger UI):** Access the interactive API documentation at `http://localhost:8000/docs`
-   **API Documentation (Redoc):** Access the alternative API documentation at `http://localhost:8000/redoc`

## Usage

### URL Anti-Fraud

Use the web dashboard to check URLs, or use `curl` to interact with the API directly. The system now checks against the AbuseIPDB database in addition to the local blocklist and heuristics.

**Check a URL:**

```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"url": "http://example.com/suspicious"}' \
     http://localhost:8000/api/url/check
```

**Add a URL to the Blocklist:**

```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"url": "http://malicious-site.com", "category": "phishing"}' \
     http://localhost:8000/api/url/blocklist
```

### Dynamic Keyword Management

Manage suspicious keywords stored in MongoDB. **After adding or removing keywords, you must restart the `fastapi-app` service for changes to take effect.**

**Add a Keyword:**

```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"keyword": "new_suspicious_term"}' \
     http://localhost:8000/api/keywords/add

# IMPORTANT: Restart the fastapi-app service after adding/removing keywords
docker compose restart fastapi-app
```

**Remove a Keyword:**

```bash
curl -X DELETE -H "Content-Type: application/json" \
     -d '{"keyword": "old_suspicious_term"}' \
     http://localhost:8000/api/keywords/remove

# IMPORTANT: Restart the fastapi-app service after adding/removing keywords
docker compose restart fastapi-app
```

**List All Keywords:**

```bash
curl http://localhost:8000/api/keywords/list
```

### Traffic Anti-Fraud

The rate-limiting is applied automatically to all API endpoints. If you make too many requests within a short period, you will receive a `429 Too Many Requests` response.

**View Recent Blocked Traffic Events:**

Check the "Recent Traffic Events" section on the web dashboard (`http://localhost:8000`) for a live feed of blocked IPs.

### Machine Learning Data Collection

**Log and Label a URL for ML Training:**

```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"url": "http://legitimate-site.com", "label": "legitimate"}' \
     http://localhost:8000/api/url/log_and_label
```

## Stopping the Application

To stop all running Docker containers and remove their networks and volumes (for MongoDB data persistence, the `mongo_data` volume will remain unless explicitly removed):

```bash
docker compose down
```

To stop and remove all containers, networks, and volumes (including MongoDB data):

```bash
docker compose down --volumes
```