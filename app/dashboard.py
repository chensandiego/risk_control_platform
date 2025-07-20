from sqlalchemy.orm import Session
from . import crud

def get_dashboard_data(db: Session):
    results = crud.get_analysis_results(db, limit=1000)  # Get the last 1000 results
    
    total_files_analyzed = len(results)
    high_risk_files = sum(1 for r in results if r.risk_score >= 150)
    medium_risk_files = sum(1 for r in results if 50 <= r.risk_score < 150)
    low_risk_files = sum(1 for r in results if r.risk_score < 50)

    risk_by_type = {}
    for r in results:
        if r.findings:
            for finding_type in r.findings.keys():
                risk_by_type[finding_type] = risk_by_type.get(finding_type, 0) + 1

    return {
        "total_files_analyzed": total_files_analyzed,
        "high_risk_files": high_risk_files,
        "medium_risk_files": medium_risk_files,
        "low_risk_files": low_risk_files,
        "risk_by_type": risk_by_type
    }
