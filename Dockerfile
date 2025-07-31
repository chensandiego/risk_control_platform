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
COPY --chown=appuser:appuser ./ner_model_v2 /home/appuser/ner_model_v2

# Switch to non-root user
USER appuser

# Expose port and run the application
EXPOSE 80
CMD ["bash", "-c", "rm -f /app/sql_app.db && mkdir -p /home/appuser/.cache/torch/hub/checkpoints && cp /app/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth /home/appuser/.cache/torch/hub/checkpoints/ && ( (PYTHONPATH=. uvicorn app.main:app --host 0.0.0.0 --port 80) & (celery -A celery_app worker --loglevel=info) )"]
