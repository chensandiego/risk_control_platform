from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from bson import ObjectId
from sqlalchemy import Column, Integer, String

from .database import Base

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, field_schema: Dict[str, Any]):
        field_schema.update(type="string")

class AnalysisResult(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id")
    filename: str
    content_type: str
    risk_score: int
    findings: Dict[str, Any]

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class CustomRule(Base):
    __tablename__ = "custom_rules"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    pattern = Column(String)
    description = Column(String)