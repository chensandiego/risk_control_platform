from sqlalchemy import Column, Integer, String, JSON

from .database import Base


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    content_type = Column(String)
    risk_score = Column(Integer)
    findings = Column(JSON)