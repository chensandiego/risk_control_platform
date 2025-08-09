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
import asyncio

from . import crud, models, schemas, rules_crud, dashboard
from .database import SessionLocal, engine, get_db
from .analysis import analyze_file_task, redact_file, quarantine_file
from .connectors import s3, google_drive, dropbox, postgresql, mysql, github, gitlab
from celery_app import celery_app

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

models.Base.metadata.create_all(bind=engine)
logger.info("Database tables created or already exist.")

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

@app.post("/scan-s3-file/")
async def scan_s3_file(scan_request: schemas.S3ScanRequest):
    """
    Triggers a scan of a file stored in an S3 bucket.
    """
    content = s3.download_s3_file(scan_request.bucket_name, scan_request.object_key)
    if isinstance(content, dict) and "error" in content:
        raise HTTPException(status_code=400, detail=content["error"])

    # Assuming the content is bytes, and we can derive a content_type
    # A more robust implementation might get the content_type from S3 metadata
    task = analyze_file_task.delay(content, "application/octet-stream", scan_request.object_key)
    return JSONResponse({"task_id": task.id})

@app.post("/scan-google-drive-file/")
async def scan_google_drive_file(scan_request: schemas.GoogleDriveScanRequest):
    """
    Triggers a scan of a file stored in Google Drive.
    """
    content = google_drive.download_drive_file(scan_request.file_id)
    if isinstance(content, dict) and "error" in content:
        raise HTTPException(status_code=400, detail=content["error"])

    # Assuming the content is bytes, and we can derive a content_type
    # A more robust implementation might get the content_type from Google Drive metadata
    task = analyze_file_task.delay(content, "application/octet-stream", scan_request.file_id)
    return JSONResponse({"task_id": task.id})

@app.post("/scan-dropbox-file/")
async def scan_dropbox_file(scan_request: schemas.DropboxScanRequest):
    """
    Triggers a scan of a file stored in Dropbox.
    """
    content = dropbox.download_dropbox_file(scan_request.file_path)
    if isinstance(content, dict) and "error" in content:
        raise HTTPException(status_code=400, detail=content["error"])

    # Assuming the content is bytes, and we can derive a content_type
    # A more robust implementation might get the content_type from Dropbox metadata
    task = analyze_file_task.delay(content, "application/octet-stream", scan_request.file_path)
    return JSONResponse({"task_id": task.id})

@app.post("/scan-postgresql-table/")
async def scan_postgresql_table(scan_request: schemas.PostgreSQLScanRequest):
    """
    Triggers a scan of a PostgreSQL table.
    """
    content = postgresql.scan_table(scan_request.table_name)
    if isinstance(content, dict) and "error" in content:
        raise HTTPException(status_code=400, detail=content["error"])

    task = analyze_file_task.delay(content.encode('utf-8'), "text/plain", scan_request.table_name)
    return JSONResponse({"task_id": task.id})

@app.post("/scan-mysql-table/")
async def scan_mysql_table(scan_request: schemas.MySQLScanRequest):
    """
    Triggers a scan of a MySQL table.
    """
    content = mysql.scan_table(scan_request.table_name)
    if isinstance(content, dict) and "error" in content:
        raise HTTPException(status_code=400, detail=content["error"])

    task = analyze_file_task.delay(content.encode('utf-8'), "text/plain", scan_request.table_name)
    return JSONResponse({"task_id": task.id})

@app.post("/scan-github-file/")
async def scan_github_file(scan_request: schemas.GitHubScanRequest):
    """
    Triggers a scan of a file in a GitHub repository.
    """
    content = github.get_repo_file_content(scan_request.repo_name, scan_request.file_path)
    if isinstance(content, dict) and "error" in content:
        raise HTTPException(status_code=400, detail=content["error"])

    task = analyze_file_task.delay(content, "application/octet-stream", scan_request.file_path)
    return JSONResponse({"task_id": task.id})

@app.post("/scan-gitlab-file/")
async def scan_gitlab_file(scan_request: schemas.GitLabScanRequest):
    """
    Triggers a scan of a file in a GitLab repository.
    """
    content = gitlab.get_project_file_content(
        scan_request.project_id, scan_request.file_path, scan_request.ref
    )
    if isinstance(content, dict) and "error" in content:
        raise HTTPException(status_code=400, detail=content["error"])

    task = analyze_file_task.delay(content, "application/octet-stream", scan_request.file_path)
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
def get_dashboard_data_endpoint():
    return dashboard.get_dashboard_data()

@app.post("/scan-docker-compose/", response_model=schemas.TrivyScanResult)
async def scan_docker_compose(scan_request: schemas.DockerComposeScanRequest):
    with open("temp_docker_compose.yml", "w") as f:
        f.write(scan_request.yaml_content)
    
    process = await asyncio.create_subprocess_shell(
        "trivy config temp_docker_compose.yml",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    
    os.remove("temp_docker_compose.yml")
    
    if process.returncode == 0:
        return {"success": True, "output": stdout.decode()}
    else:
        return {"success": False, "output": stderr.decode()}

@app.post("/scan-dockerfile/", response_model=schemas.TrivyScanResult)
async def scan_dockerfile(scan_request: schemas.DockerfileScanRequest):
    # Create a temporary file for the Dockerfile content
    temp_dockerfile_path = "temp_dockerfile"
    with open(temp_dockerfile_path, "w") as f:
        f.write(scan_request.dockerfile_content)
    
    # Run Trivy to scan the Dockerfile for vulnerabilities and misconfigurations
    # Using 'fs' subcommand for local file system scan, and '--security-checks vuln,config,secret'
    # to include vulnerability, misconfiguration, and secret scanning.
    process = await asyncio.create_subprocess_shell(
        f"trivy fs --scanners vuln,misconfig,secret --file-patterns \"dockerfile:temp_dockerfile\" --format json {temp_dockerfile_path}",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    
    # Clean up the temporary Dockerfile
    os.remove(temp_dockerfile_path)
    
    if process.returncode == 0:
        return {"success": True, "output": stdout.decode()}
    else:
        # If Trivy exits with a non-zero code, it might indicate issues found or an error.
        # We return the stderr as output for debugging in case of an error.
        return {"success": False, "output": stderr.decode()}