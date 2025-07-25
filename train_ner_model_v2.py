

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import numpy as np
import json

# --- 1. Configuration ---
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "./ner_model_v2"
LOGGING_DIR = "./ner_logs_v2"
DATA_FILE = "./ner_data.jsonl"

# --- 2. Define Your Labels ---
labels_list = ["O", "B-PII", "I-PII"]
label_map = {label: i for i, label in enumerate(labels_list)}

# --- 3. Load Your Dataset ---
def load_data(file_path):
    texts, labels = [], []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            texts.append(data['tokens'])
            labels.append(data['ner_tags'])
    return texts, labels

print(f"Loading data from {DATA_FILE}...")
texts, labels = load_data(DATA_FILE)

# --- 4. Split Data into Training and Validation Sets ---
# A validation set is crucial for monitoring performance and preventing overfitting.
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# --- 5. Load the Tokenizer ---
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# --- 6. Tokenize and Align Labels ---
def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            current_word = word_id
            label = -100 if word_id is None else label_map[labels[word_id]]
            new_labels.append(label)
        elif word_id is None:
            new_labels.append(-100)
        else:
            label = label_map[labels[word_id]]
            if label == label_map["B-PII"]:
                new_labels.append(label_map["I-PII"])
            else:
                new_labels.append(label)
    return new_labels

def tokenize_and_align(texts, labels):
    encodings = tokenizer(texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
    aligned_labels = []
    for i, text_labels in enumerate(labels):
        word_ids = encodings.word_ids(i)
        aligned_labels.append(align_labels_with_tokens(text_labels, word_ids))
    encodings.pop("offset_mapping")
    encodings["labels"] = aligned_labels
    return encodings

print("Tokenizing and aligning labels for training and validation sets...")
train_encodings = tokenize_and_align(train_texts, train_labels)
val_encodings = tokenize_and_align(val_texts, val_labels)

# --- 7. Create PyTorch Datasets ---
class NERDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.labels)

train_dataset = NERDataset(train_encodings)
val_dataset = NERDataset(val_encodings)

# --- 8. Load the Pre-trained Model ---
print("Loading pre-trained model...")
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels_list),
    id2label={i: label for i, label in enumerate(labels_list)},
    label2id=label_map
)

# --- 9. Define Training Arguments ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir=LOGGING_DIR,
    logging_steps=10,
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    save_strategy="epoch",        # Save the model at the end of each epoch
    load_best_model_at_end=True,  # Load the best model at the end of training
)

# --- 10. Define Evaluation Metrics ---
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [labels_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [labels_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
        "classification_report": classification_report(true_labels, true_predictions),
    }

# --- 11. Initialize the Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# --- 11. Start Training ---
print("Starting model training...")
trainer.train()
print("Training finished!")

# --- 12. Save the Best Model and Tokenizer ---
print(f"Saving the best model and tokenizer to {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n--- Training Complete ---")
print(f"You can now find your fine-tuned model in the '{OUTPUT_DIR}' directory.")
