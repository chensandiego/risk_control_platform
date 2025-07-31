# Dockerfile
FROM python:3.9-slim

# Update and upgrade packages
RUN apt-get update && apt-get upgrade -y

# Create a non-root user
RUN useradd --create-home appuser
WORKDIR /home/appuser

# Copy and install dependencies
COPY --chown=appuser:appuser ./requirements.txt ./
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy application code
COPY --chown=appuser:appuser ./app ./app

# Switch to non-root user
USER appuser

# Expose port and run the application
EXPOSE 80
CMD ["bash", "-c", "PYTHONPATH=. uvicorn app.main:app --host 0.0.0.0 --port 80"]
