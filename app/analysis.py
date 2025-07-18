import re

# Define sensitive data patterns and their associated risk weights
SENSITIVE_PATTERNS = {
    "email_addresses": {
        "pattern": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        "weight": 5,
        "description": "Potential email addresses found."
    },
    "credit_card_numbers": {
        "pattern": r'\b(?:\d[ -]*?){13,16}\b',  # Basic pattern, needs more robust validation for real-world use
        "weight": 10,
        "description": "Potential credit card numbers found."
    },
    "api_keys": {
        "pattern": r"""(?:api_key|API_KEY|token|bearer|secret)[\s=:]*['"]?([a-zA-Z0-9-_\.]{16,})['"]?""",
        "weight": 15,
        "description": "Potential API keys or tokens found."
    },
    "social_security_numbers": {
        "pattern": r'\b\d{3}-\d{2}-\d{4}\b',
        "weight": 20,
        "description": "Potential Social Security Numbers found."
    },
    "private_keys": {
        "pattern": r'-----BEGIN (RSA|DSA|EC|PGP) PRIVATE KEY-----',
        "weight": 25,
        "description": "Potential private cryptographic keys found."
    }
}

def analyze_file(content: bytes):
    content_str = content.decode('utf-8', errors='ignore')
    
    total_risk_score = 0
    findings = {}
    
    for category, data in SENSITIVE_PATTERNS.items():
        matches = re.findall(data["pattern"], content_str)
        if matches:
            findings[category] = {
                "count": len(matches),
                "matches": matches,
                "description": data["description"],
                "risk_contribution": len(matches) * data["weight"]
            }
            total_risk_score += len(matches) * data["weight"]

    # Basic Anomaly Detection Placeholder:
    # This is a very simple example. Real anomaly detection would involve
    # statistical analysis, machine learning, or predefined rules based on
    # expected content/behavior.
    anomalies = []
    if len(content_str) > 100000: # Example: Flag very large files as potential anomalies
        anomalies.append("File size is unusually large (over 100KB).")
        total_risk_score += 50 # Add a fixed score for this anomaly

    if anomalies:
        findings["anomalies"] = {
            "count": len(anomalies),
            "matches": anomalies,
            "description": "Potential anomalies detected.",
            "risk_contribution": 50 # Fixed contribution for this example
        }

    return {
        "overall_risk_score": total_risk_score,
        "detailed_findings": findings,
        "summary": generate_risk_summary(total_risk_score, findings)
    }

def generate_risk_summary(score: int, findings: dict):
    summary_lines = [f"Overall Risk Score: {score}"]
    
    if score == 0:
        summary_lines.append("No significant risks detected.")
    elif score < 50:
        summary_lines.append("Low risk: Minor issues found.")
    elif score < 150:
        summary_lines.append("Medium risk: Some sensitive data or anomalies detected.")
    else:
        summary_lines.append("High risk: Significant sensitive data or critical anomalies detected. Immediate review recommended.")

    if findings:
        summary_lines.append("\nDetailed Findings:")
        for category, data in findings.items():
            summary_lines.append(f"- {data['description']} (Count: {data['count']}, Risk Contribution: {data['risk_contribution']})")
            # Optionally, add specific matches for review, but be careful with sensitive data in logs
            # For a real report, you might redact or only show partial matches
            # summary_lines.append(f"  Matches: {data['matches']}") 
    
    return "\n".join(summary_lines)

# Example usage (for testing purposes, not part of the main app flow)
if __name__ == "__main__":
    sample_content_1 = b"This is a test file with an email: test@example.com and a credit card: 1234-5678-9012-3456."
    analysis_result_1 = analyze_file(sample_content_1)
    print("--- Analysis Result 1 ---")
    print(analysis_result_1["summary"])
    print("\n")

    sample_content_2 = b"No sensitive data here. Just some random text."
    analysis_result_2 = analyze_file(sample_content_2)
    print("--- Analysis Result 2 ---")
    print(analysis_result_2["summary"])
    print("\n")

    sample_content_3 = b"API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    analysis_result_3 = analyze_file(sample_content_3)
    print("--- Analysis Result 3 ---")
    print(analysis_result_3["summary"])
    print("\n")
    
    # Large file content for anomaly detection test
    sample_content_4 = b"a" * 100001 
    analysis_result_4 = analyze_file(sample_content_4)
    print("--- Analysis Result 4 (Large File) ---")
    print(analysis_result_4["summary"])
    print("\n")