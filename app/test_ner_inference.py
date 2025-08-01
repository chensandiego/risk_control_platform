import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import time
import os

ner_model_path = "/home/appuser/ner_model_v2"

print(f"[{time.time()}] Attempting to load NER tokenizer from {ner_model_path}...")
try:
    ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_path)
    print(f"[{time.time()}] Tokenizer loaded successfully.")
except Exception as e:
    print(f"[{time.time()}] Error loading tokenizer: {e}")
    exit(1)

print(f"[{time.time()}] Attempting to load NER model from {ner_model_path}...")
try:
    ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_path)
    ner_model.to('cpu')
    ner_model.eval()
    print(f"[{time.time()}] Model loaded and set to CPU successfully.")
except Exception as e:
    print(f"[{time.time()}] Error loading model: {e}")
    exit(1)

# Test inference with a simple text
test_text = "This is a test sentence for NER."
print(f"[{time.time()}] Testing inference with text: '{test_text}'")

try:
    inputs = ner_tokenizer(test_text, return_tensors="pt", truncation=True).to('cpu')
    print(f"[{time.time()}] Input tensor created. Input IDs shape: {inputs['input_ids'].shape}")

    with torch.no_grad():
        print(f"[{time.time()}] Before model inference call.")
        outputs = ner_model(**inputs)
        print(f"[{time.time()}] After model inference call. Output shape: {outputs.logits.shape}")

    predictions = torch.argmax(outputs.logits, dim=2)
    print(f"[{time.time()}] Inference successful. Predictions shape: {predictions.shape}")

    # Decode predictions (simplified for testing)
    label_map = ner_model.config.id2label
    ner_results = []
    for token, prediction in zip(inputs.tokens(), predictions[0].tolist()):
        entity_type = label_map[prediction]
        ner_results.append({"word": token, "entity_type": entity_type})
    print(f"[{time.time()}] Extracted entities: {ner_results}")

except Exception as e:
    print(f"[{time.time()}] Error during inference: {e}")
    exit(1)

print(f"[{time.time()}] NER inference test completed.")
