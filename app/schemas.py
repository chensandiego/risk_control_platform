from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any, List
from app.models import PyObjectId
from datetime import datetime

class AnalysisResultBase(BaseModel):
    filename: str
    content_type: Optional[str] = None
    risk_score: int
    findings: Optional[Dict[str, Any]] = None

class AnalysisResultCreate(AnalysisResultBase):
    pass

class AnalysisResult(AnalysisResultBase):
    id: PyObjectId = Field(alias="_id")

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True, json_encoders={PyObjectId: str})

class TextInput(BaseModel):
    text: str

class CustomRuleBase(BaseModel):
    name: str
    pattern: str
    description: Optional[str] = None
    version: int = 1
    created_at: datetime = Field(default_factory=datetime.utcnow)

class CustomRuleCreate(CustomRuleBase):
    pass

class CustomRuleUpdate(CustomRuleBase):
    pass

class CustomRule(CustomRuleBase):
    id: int

    model_config = ConfigDict(from_attributes=True)

class RuleTestRequest(BaseModel):
    pattern: str
    text: str

class RuleImportRequest(BaseModel):
    rules: List[CustomRuleCreate]
