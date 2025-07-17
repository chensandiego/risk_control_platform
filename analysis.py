import re

def analyze_file(content: bytes):
    content_str = content.decode('utf-8', errors='ignore')
    findings = {}
    risk_score = 0

    # Example: Scan for email addresses
    emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', content_str)
    if emails:
        findings['emails'] = emails
        risk_score += len(emails) * 5

    # Example: Scan for credit card numbers (very basic)
    credit_cards = re.findall(r'\b(?:\d[ -]*?){13,16}\b', content_str)
    if credit_cards:
        findings['credit_cards'] = credit_cards
        risk_score += len(credit_cards) * 10

    return {"risk_score": risk_score, "findings": findings}