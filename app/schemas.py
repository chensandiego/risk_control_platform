from pydantic import BaseModel
from typing import Optional, Dict, Any

class AnalysisResultBase(BaseModel):
    filename: str
    content_type: Optional[str] = None
    risk_score: int
    findings: Optional[Dict[str, Any]] = None

class AnalysisResultCreate(AnalysisResultBase):
    pass

class AnalysisResult(AnalysisResultBase):
    id: int

    class Config:
        orm_mode = True