import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

print("Starting NER model test...")

ner_model_path = "/home/appuser/ner_model_v2"

try:
    print(f"Attempting to load tokenizer from {ner_model_path}")
    ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_path)
    print("Tokenizer loaded successfully.")

    print(f"Attempting to load model from {ner_model_path}")
    ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_path)
    print("Model loaded successfully.")

    print("Creating NER pipeline...")
    ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, device="cpu")
    print("NER pipeline created successfully.")

    test_text = "My name is John Doe and I live in New York."
    print(f"Testing NER pipeline with text: '{test_text}'")
    results = ner_pipeline(test_text)
    print("NER pipeline executed successfully with test text.")
    print("Results:")
    for entity in results:
        print(f"  Word: {entity['word']}, Entity Type: {entity['entity']}")

except Exception as e:
    print(f"An error occurred during NER test: {e}")

print("NER model test finished.")