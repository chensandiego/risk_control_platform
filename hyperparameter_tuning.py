
import torch
import optuna
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import numpy as np
import json
from seqeval.metrics import f1_score

# --- 1. Configuration ---
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "./ner_model_hp"
LOGGING_DIR = "./ner_logs_hp"
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

texts, labels = load_data(DATA_FILE)

# --- 4. Split Data ---
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# --- 5. Tokenizer and Data Alignment ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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

train_encodings = tokenize_and_align(train_texts, train_labels)
val_encodings = tokenize_and_align(val_texts, val_labels)

class NERDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.labels)

train_dataset = NERDataset(train_encodings)
val_dataset = NERDataset(val_encodings)

# --- 6. Model Initializer ---
def model_init():
    return AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels_list),
        id2label={i: label for i, label in enumerate(labels_list)},
        label2id=label_map
    )

# --- 7. Objective Function for Optuna ---
def objective(trial):
    training_args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/trial_{trial.number}",
        logging_dir=f"{LOGGING_DIR}/trial_{trial.number}",
        num_train_epochs=trial.suggest_int("num_train_epochs", 2, 4),
        per_device_train_batch_size=trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16]),
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        weight_decay=trial.suggest_float("weight_decay", 0.0, 0.1),
        warmup_steps=trial.suggest_int("warmup_steps", 0, 500),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to=["none"], # Disable wandb or other integrations
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    eval_results = trainer.evaluate()
    return eval_results["eval_loss"]

# --- 8. Run Hyperparameter Search ---
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10) # Run 10 trials

# --- 9. Print and Save Best Hyperparameters ---
print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Save the best hyperparameters to a file
best_params = trial.params
with open("best_hyperparameters.json", "w") as f:
    json.dump(best_params, f)

print("\n--- Hyperparameter Tuning Complete ---")
print("Best hyperparameters saved to best_hyperparameters.json")
