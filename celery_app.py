
from celery import Celery
import os

# Default to a local Redis instance if the environment variable is not set
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "tasks",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["app.analysis"]  # Point to the module where tasks are defined
)

# Optional: Configure Celery for better performance and reliability
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
enable_utc=True,
)
