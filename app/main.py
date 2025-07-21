from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from typing import List
import re
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
import redis
import os

from . import crud, models, schemas, rules_crud, dashboard
from .database import SessionLocal, engine, get_db
from .analysis import analyze_file_task, redact_file, quarantine_file
from celery_app import celery_app

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Initialize Redis client
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.from_url(REDIS_URL)

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    content = await file.read()
    task = analyze_file_task.delay(content, file.content_type, file.filename)
    return JSONResponse({"task_id": task.id})

@app.post("/analyze-text/")
async def analyze_text_directly(text_data: schemas.TextInput):
    task = analyze_file_task.delay(text_data.text.encode('utf-8'), "text/plain", "Direct Text Input")
    return JSONResponse({"task_id": task.id})

@app.post("/remediate/{task_id}")
async def remediate_file(task_id: str, action: str):
    task_result = AsyncResult(task_id, app=celery_app)
    if not task_result.ready() or not task_result.successful():
        raise HTTPException(status_code=404, detail="Task not found or not completed")

    result = task_result.get()
    filename = result.get("filename")
    content = result.get("original_content")
    findings = result.get("detailed_findings")

    if action == "redact":
        redacted_content = redact_file(content.encode('utf-8'), findings)
        return FileResponse(io.BytesIO(redacted_content), media_type="application/octet-stream", filename=f"redacted_{filename}")
    elif action == "quarantine":
        quarantine_file(content.encode('utf-8'), filename)
        return {"message": f"File {filename} has been quarantined."}
    else:
        raise HTTPException(status_code=400, detail="Invalid remediation action")

@app.get("/results/{task_id}")
async def get_analysis_result(task_id: str):
    # Try to fetch from cache first
    cached_result = redis_client.get(task_id)
    if cached_result:
        return JSONResponse({"status": "SUCCESS", "result": json.loads(cached_result)})

    task_result = AsyncResult(task_id, app=celery_app)

    if task_result.ready():
        if task_result.successful():
            analysis_data = task_result.get()
            # Cache the result before returning
            redis_client.set(task_id, json.dumps(analysis_data), ex=3600) # Cache for 1 hour
            return JSONResponse({"status": "SUCCESS", "result": analysis_data})
        else:
            return JSONResponse({"status": "FAILURE", "error": str(task_result.info)})
    else:
        return JSONResponse({"status": "PENDING"})

@app.get("/", response_class=HTMLResponse)
async def main():
    return FileResponse('app/static/index.html')

@app.post("/rules/test/")
async def test_rule(request: schemas.RuleTestRequest):
    try:
        re.compile(request.pattern)
    except re.error:
        raise HTTPException(status_code=400, detail="Invalid regex pattern")
    
    matches = re.findall(request.pattern, request.text)
    return {"matches": matches}

@app.post("/rules/import/")
async def import_rules(request: schemas.RuleImportRequest, db: Session = Depends(get_db)):
    imported_count = 0
    for rule_data in request.rules:
        existing_rule = db.query(models.CustomRule).filter(models.CustomRule.name == rule_data.name).first()
        if existing_rule:
            # For simplicity, we'll update the existing rule. A more sophisticated approach might handle versioning or conflicts.
            rules_crud.update_rule(db, existing_rule.id, rule_data)
        else:
            rules_crud.create_rule(db, rule_data)
        imported_count += 1
    return {"message": f"Successfully imported {imported_count} rules."}

@app.get("/rules/export/", response_model=List[schemas.CustomRule])
async def export_rules(db: Session = Depends(get_db)):
    rules = rules_crud.get_rules(db)
    return rules

@app.post("/rules/", response_model=schemas.CustomRule)
def create_rule(rule: schemas.CustomRuleCreate, db: Session = Depends(get_db)):
    return rules_crud.create_rule(db=db, rule=rule)

@app.get("/rules/", response_model=list[schemas.CustomRule])
def read_rules(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    rules = rules_crud.get_rules(db, skip=skip, limit=limit)
    return rules

@app.get("/rules/{rule_id}", response_model=schemas.CustomRule)
def read_rule(rule_id: int, db: Session = Depends(get_db)):
    db_rule = rules_crud.get_rule(db, rule_id=rule_id)
    if db_rule is None:
        raise HTTPException(status_code=404, detail="Rule not found")
    return db_rule

@app.put("/rules/{rule_id}", response_model=schemas.CustomRule)
def update_rule(rule_id: int, rule: schemas.CustomRuleUpdate, db: Session = Depends(get_db)):
    db_rule = rules_crud.update_rule(db, rule_id=rule_id, rule=rule)
    if db_rule is None:
        raise HTTPException(status_code=404, detail="Rule not found")
    return db_rule

@app.delete("/rules/{rule_id}", response_model=schemas.CustomRule)
def delete_rule(rule_id: int, db: Session = Depends(get_db)):
    db_rule = rules_crud.delete_rule(db, rule_id=rule_id)
    if db_rule is None:
        raise HTTPException(status_code=44, detail="Rule not found")
    return db_rule

@app.get("/dashboard/")
def get_dashboard_data(db: Session = Depends(get_db)):
    return dashboard.get_dashboard_data(db)