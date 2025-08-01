# Dockerfile
FROM python:3.9-slim

# Update and upgrade packages
RUN apt-get update && apt-get upgrade -y && apt-get install -y wget

# Create a non-root user
RUN useradd --create-home appuser
WORKDIR /home/appuser

# Copy and install dependencies
COPY --chown=appuser:appuser ./requirements.txt ./
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy application code
COPY --chown=appuser:appuser ./app ./app
COPY --chown=appuser:appuser ./ner_model_v2 /home/appuser/ner_model_v2
COPY --chown=appuser:appuser ./fasterrcnn_resnet50_fpn_coco-258fb6c6.pth /home/appuser/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth

# Switch to non-root user
USER appuser

# Set TORCH_HOME to ensure models are loaded from the pre-populated cache
ENV TORCH_HOME=/home/appuser/.cache/torch/hub

# Expose port and run the application
EXPOSE 80
RUN mkdir -p /home/appuser/.cache/torch/hub/checkpoints &&     wget -O /home/appuser/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth https://download.pytorch.org/models/resnet50-0676ba61.pth &&     cp /home/appuser/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth /home/appuser/.cache/torch/hub/checkpoints/
CMD ["bash", "-c", "rm -f /app/sql_app.db && ( (PYTHONPATH=. uvicorn app.main:app --host 0.0.0.0 --port 80) & (celery -A celery_app worker --loglevel=info) )"]
