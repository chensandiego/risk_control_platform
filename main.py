from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from . import crud, models, schemas, analysis
from .database import SessionLocal, engine

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/uploadfile/", response_model=schemas.AnalysisResult)
async def create_upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file:
        raise HTTPException(status_code=400, detail="No file part")

    content = await file.read()
    analysis_results = analysis.analyze_file(content)

    result = schemas.AnalysisResultCreate(
        filename=file.filename,
        content_type=file.content_type,
        risk_score=analysis_results['risk_score'],
        findings=analysis_results['findings']
    )
    return crud.create_analysis_result(db=db, result=result)

@app.get("/", response_class=HTMLResponse)
async def main():
    return FileResponse('app/static/index.html')