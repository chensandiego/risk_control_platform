
import csv
import io
from fpdf import FPDF
from typing import List, Dict, Any

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Analysis Report', 0, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf_report(results: List[Dict[str, Any]]) -> bytes:
    pdf = PDF()
    pdf.add_page()
    pdf.set_font('Arial', size=10)

    for result in results:
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, f"Filename: {result.get('filename', 'N/A')}", 0, 1)
        pdf.set_font('Arial', size=10)
        pdf.cell(0, 10, f"Content-Type: {result.get('content_type', 'N/A')}", 0, 1)
        pdf.cell(0, 10, f"Risk Score: {result.get('risk_score', 'N/A')}", 0, 1)
        pdf.multi_cell(0, 10, f"Summary: {result.get('summary', 'N/A')}")
        pdf.ln(10)

    return pdf.output(dest='S').encode('latin-1')

def generate_csv_report(results: List[Dict[str, Any]]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(["Filename", "Content-Type", "Risk Score", "Summary"])

    for result in results:
        writer.writerow([
            result.get('filename', 'N/A'),
            result.get('content_type', 'N/A'),
            result.get('risk_score', 'N/A'),
            result.get('summary', 'N/A')
        ])

    return output.getvalue()
