from pymongo import MongoClient
import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

MONGO_DETAILS = os.getenv("MONGO_DETAILS", "mongodb://localhost:27017")

client = MongoClient(MONGO_DETAILS)
database = client.risk_control_db

def get_collection(collection_name: str):
    return database[collection_name]

# SQLAlchemy setup for SQLite
SQLALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()