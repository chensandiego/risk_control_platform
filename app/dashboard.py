from collections import Counter
from datetime import datetime, timedelta
from . import crud

def get_dashboard_data():
    """Gathers and processes data for the dashboard."""
    results = crud.get_all_analysis_results()
    
    # --- Existing Metrics ---
    total_files_analyzed = len(results)
    
    risk_distribution = Counter(r.get("risk_level", "Unknown") for r in results)
    
    findings_by_type = Counter()
    for r in results:
        if r.get("findings"):
            for finding in r["findings"]:
                findings_by_type[finding["type"]] += 1

    # --- New: Files Analyzed Over Time (Last 14 Days) ---
    analysis_by_day = Counter()
    fourteen_days_ago = datetime.now() - timedelta(days=14)
    for r in results:
        # Ensure timestamp is a datetime object
        timestamp_str = r.get("timestamp")
        if isinstance(timestamp_str, str):
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                if timestamp > fourteen_days_ago:
                    day = timestamp.strftime("%Y-%m-%d")
                    analysis_by_day[day] += 1
            except ValueError:
                continue # Skip if the timestamp format is incorrect

    # Sort by date and format for the chart
    sorted_analysis_by_day = sorted(analysis_by_day.items())
    time_series_labels = [date for date, count in sorted_analysis_by_day]
    time_series_data = [count for date, count in sorted_analysis_by_day]

    # --- New: Recent High-Risk Files ---
    recent_high_risk_files = crud.get_recent_high_risk_files(limit=5)

    return {
        "total_files_analyzed": total_files_analyzed,
        "risk_distribution": dict(risk_distribution),
        "findings_by_type": dict(findings_by_type),
        "time_series": {
            "labels": time_series_labels,
            "data": time_series_data
        },
        "recent_high_risk_files": [
            {
                "filename": r.get("filename", "N/A"),
                "timestamp": r.get("timestamp", "N/A")
            }
            for r in recent_high_risk_files
        ]
    }
