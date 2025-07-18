from fastapi.testclient import TestClient
from app.main import app
from unittest.mock import patch
import os

client = TestClient(app)

def test_upload_file():
    with patch('app.main.analyze_file_task.delay') as mock_delay:
        mock_delay.return_value.id = "mock_task_id"
        with open("sample_data.txt", "rb") as f:
            response = client.post("/uploadfile/", files={"file": ("sample_data.txt", f, "text/plain")})
        assert response.status_code == 200
        assert "task_id" in response.json()
        mock_delay.assert_called_once()

def test_upload_image_with_ocr():
    # Create a dummy image file for testing
    from PIL import Image, ImageDraw, ImageFont
    img_path = "test_image.png"
    img = Image.new('RGB', (200, 50), color = (255, 255, 255))
    d = ImageDraw.Draw(img)
    # Try to use a default font or handle font loading errors
    try:
        font = ImageFont.truetype("arial.ttf", 20) # Common font on Windows/macOS
    except IOError:
        font = ImageFont.load_default() # Fallback to default PIL font
    d.text((10,10), "test@example.com", fill=(0,0,0), font=font)
    img.save(img_path)

    with patch('app.main.analyze_file_task.delay') as mock_delay:
        mock_delay.return_value.id = "mock_image_task_id"
        with open(img_path, "rb") as f:
            response = client.post("/uploadfile/", files={"file": (img_path, f, "image/png")})
        assert response.status_code == 200
        assert "task_id" in response.json()
        mock_delay.assert_called_once()
    
    os.remove(img_path)

def test_analyze_text_with_ner():
    text_input = {"text": "John Doe works at Google in New York."}
    response = client.post("/analyze-text/", json=text_input)
    assert response.status_code == 200
    data = response.json()
    assert data["risk_score"] > 0
    assert "person_names" in data["findings"]
    assert "locations" in data["findings"]
    assert "organizations" in data["findings"]
    assert "John Doe" in data["findings"]["person_names"]["matches"]
    assert "New York" in data["findings"]["locations"]["matches"]
    assert "Google" in data["findings"]["organizations"]["matches"]
