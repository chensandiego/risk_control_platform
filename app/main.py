from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
import io
from docx import Document
import openpyxl
import pytesseract
from PIL import Image
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage

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

def extract_text_from_docx(file_content: bytes) -> str:
    doc = Document(io.BytesIO(file_content))
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                full_text.append(cell.text)
    return "\n".join(full_text)

def extract_text_from_xlsx(file_content: bytes) -> str:
    workbook = openpyxl.load_workbook(io.BytesIO(file_content))
    full_text = []
    for sheet_name in workbook.sheetnames:
        sheet = workbook[sheet_name]
        for row in sheet.iter_rows():
            for cell in row:
                if cell.value is not None:
                    full_text.append(str(cell.value))
    return "\n".join(full_text)

def extract_text_from_pdf(file_content: bytes) -> str:
    rsrcmgr = PDFResourceManager()
    retstr = io.StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    fp = io.BytesIO(file_content)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    for page in PDFPage.get_pages(fp):
        interpreter.process_page(page)
    text = retstr.getvalue()
    fp.close()
    device.close()
    retstr.close()
    return text

def extract_text_from_image(file_content: bytes) -> str:
    image = Image.open(io.BytesIO(file_content))
    text = pytesseract.image_to_string(image)
    return text

@app.post("/uploadfile/", response_model=schemas.AnalysisResult)
async def create_upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file:
        raise HTTPException(status_code=400, detail="No file part")

    content = await file.read()
    extracted_text = ""

    if file.content_type == "text/plain":
        extracted_text = content.decode('utf-8')
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

    analysis_results = analysis.analyze_file(extracted_text.encode('utf-8')) # analysis.analyze_file expects bytes

    result = schemas.AnalysisResultCreate(
        filename=file.filename,
        content_type=file.content_type,
        risk_score=analysis_results['overall_risk_score'],
        findings=analysis_results['detailed_findings']
    )
    return crud.create_analysis_result(db=db, result=result)

@app.post("/analyze-text/", response_model=schemas.AnalysisResult)
async def analyze_text_directly(text_data: schemas.TextInput, db: Session = Depends(get_db)):
    extracted_text = text_data.text
    analysis_results = analysis.analyze_file(extracted_text.encode('utf-8'))

    result = schemas.AnalysisResultCreate(
        filename="Direct Text Input",
        content_type="text/plain",
        risk_score=analysis_results['overall_risk_score'],
        findings=analysis_results['detailed_findings']
    )
    return crud.create_analysis_result(db=db, result=result)

@app.get("/", response_class=HTMLResponse)
async def main():
    return FileResponse('app/static/index.html')