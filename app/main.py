from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from celery.result import AsyncResult
import io
import csv
import json
import xml.etree.ElementTree as ET
from docx import Document
import openpyxl
import pytesseract
from PIL import Image
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage

from . import crud, models, schemas
from .database import SessionLocal, engine
from .analysis import analyze_file_task
from celery_app import celery_app

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

def extract_text_from_csv(file_content: bytes) -> str:
    decoded_content = file_content.decode('utf-8').splitlines()
    reader = csv.reader(decoded_content)
    return "\n".join([",".join(row) for row in reader])

def extract_text_from_json(file_content: bytes) -> str:
    return json.dumps(json.loads(file_content.decode('utf-8')))

def extract_text_from_xml(file_content: bytes) -> str:
    root = ET.fromstring(file_content)
    return ET.tostring(root, encoding='unicode', method='text')

def extract_text_from_docx(file_content: bytes) -> str:
    doc = Document(io.BytesIO(file_content))
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_xlsx(file_content: bytes) -> str:
    workbook = openpyxl.load_workbook(io.BytesIO(file_content))
    return "\n".join([str(cell.value) for row in workbook.active.iter_rows() for cell in row if cell.value is not None])

def extract_text_from_pdf(file_content: bytes) -> str:
    # ... (implementation remains the same)
    return ""

def extract_text_from_image(file_content: bytes) -> str:
    image = Image.open(io.BytesIO(file_content))
    return pytesseract.image_to_string(image)

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    content = await file.read()
    extracted_text = ""

    if file.content_type == "text/plain":
        extracted_text = content.decode('utf-8')
    elif file.content_type == "text/csv":
        extracted_text = extract_text_from_csv(content)
    elif file.content_type == "application/json":
        extracted_text = extract_text_from_json(content)
    elif file.content_type == "application/xml" or file.content_type == "text/xml":
        extracted_text = extract_text_from_xml(content)
    elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        extracted_text = extract_text_from_docx(content)
    elif file.content_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        extracted_text = extract_text_from_xlsx(content)
    elif file.content_type == "application/pdf":
        extracted_text = extract_text_from_pdf(content)
    elif file.content_type.startswith("image/"):
        extracted_text = extract_text_from_image(content)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    task = analyze_file_task.delay(extracted_text.encode('utf-8'), file.content_type)
    
    return JSONResponse({"task_id": task.id})

@app.post("/analyze-text/")
async def analyze_text_directly(text_data: schemas.TextInput):
    task = analyze_file_task.delay(text_data.text.encode('utf-8'))
    return JSONResponse({"task_id": task.id})

@app.get("/results/{task_id}")
async def get_analysis_result(task_id: str, db: Session = Depends(get_db)):
    task_result = AsyncResult(task_id, app=celery_app)

    if task_result.ready():
        if task_result.successful():
            analysis_data = task_result.get()
            result_to_save = schemas.AnalysisResultCreate(
                filename="N/A",
                content_type="N/A",
                risk_score=analysis_data['overall_risk_score'],
                findings=analysis_data['detailed_findings']
            )
            crud.create_analysis_result(db=db, result=result_to_save)
            return JSONResponse({"status": "SUCCESS", "result": analysis_data})
        else:
            return JSONResponse({"status": "FAILURE", "error": str(task_result.info)})
    else:
        return JSONResponse({"status": "PENDING"})

@app.get("/", response_class=HTMLResponse)
async def main():
    return FileResponse('app/static/index.html')