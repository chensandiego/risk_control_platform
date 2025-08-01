import pytest
from app.analysis import analyze_file_task, shannon_entropy, analyze_image_content, analyze_text_with_ner
from PIL import Image
import io

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

def test_analyze_text_with_ner():
    text = "John Doe lives in New York City."
    entities = analyze_text_with_ner(text)
    assert len(entities) > 0
    # You can add more specific assertions based on expected NER output
    # For example, checking for specific entity types or words
    found_john = False
    found_new_york = False
    for entity in entities:
        if entity["word"] == "John" and entity["entity_type"] == "B-PER":
            found_john = True
        if entity["word"] == "New York" and entity["entity_type"] == "B-LOC":
            found_new_york = True
    assert found_john or found_new_york # At least one of these should be found, depending on model performance
