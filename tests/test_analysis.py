import pytest
from app.main import extract_text_from_csv, extract_text_from_json, extract_text_from_xml
from app.analysis import analyze_file_task, shannon_entropy, analyze_image_content
from PIL import Image
import io

def test_extract_text_from_csv():
    content = b"header1,header2\nvalue1,value2"
    expected_text = "header1,header2\nvalue1,value2"
    assert extract_text_from_csv(content) == expected_text

def test_extract_text_from_json():
    content = b'{"key": "value", "nested": {"key2": "value2"}}'
    expected_text = '{"key": "value", "nested": {"key2": "value2"}}'
    assert extract_text_from_json(content) == expected_text

def test_extract_text_from_xml():
    content = b'<root><element>value</element></root>'
    expected_text = 'value'
    assert extract_text_from_xml(content).strip() == expected_text

def test_shannon_entropy():
    assert shannon_entropy("hello") > 1.9
    assert shannon_entropy("aaaaaaaa") == 0

def test_analyze_image_content():
    # Create a dummy image
    image = Image.new('RGB', (100, 100), color = 'red')
    byte_arr = io.BytesIO()
    image.save(byte_arr, format='PNG')
    image_bytes = byte_arr.getvalue()

    # This test will be slow as it loads the model.
    # In a real-world scenario, you might mock the model.
    detected_objects = analyze_image_content(image_bytes)
    # Since it's a blank red image, we don't expect to detect anything from the COCO dataset.
    assert len(detected_objects) == 0
