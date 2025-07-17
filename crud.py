from sqlalchemy.orm import Session

from . import models, schemas


def get_analysis_result(db: Session, result_id: int):
    return db.query(models.AnalysisResult).filter(models.AnalysisResult.id == result_id).first()


def get_analysis_results(db: Session, skip: int = 0, limit: int = 10):
    return db.query(models.AnalysisResult).offset(skip).limit(limit).all()


def create_analysis_result(db: Session, result: schemas.AnalysisResultCreate):
    db_result = models.AnalysisResult(
        filename=result.filename,
        content_type=result.content_type,
        risk_score=result.risk_score,
        findings=result.findings
    )
    db.add(db_result)
    db.commit()
    db.refresh(db_result)
    return db_result