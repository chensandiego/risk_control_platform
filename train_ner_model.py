# train_ner_model.py

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import numpy as np

# --- 1. Configuration ---
MODEL_NAME = "distilbert-base-uncased"
# The directory where the fine-tuned model will be saved.
OUTPUT_DIR = "./ner_model"
# The directory where training logs and checkpoints will be saved.
LOGGING_DIR = "./ner_logs"

# --- 2. Define Your Labels ---
# Define the custom labels (tags) for your Named Entity Recognition (NER) task.
# O - Outside: The token is not part of any sensitive data entity.
# B-PII - Beginning: The token is the beginning of a Personally Identifiable Information entity.
# I-PII - Inside: The token is inside a PII entity (but not the first token).
labels_list = ["O", "B-PII", "I-PII"]
label_map = {label: i for i, label in enumerate(labels_list)}

# --- 3. Prepare Your Dataset ---
# In a real-world scenario, you would load this data from a file (e.g., CSV, JSON).
# This is a small, hardcoded example to demonstrate the process.

# Each list of words represents a sentence or a piece of text.
texts = [
    ["My", "name", "is", "John", "Doe", "and", "my", "email", "is", "john.doe@email.com"],
    ["Please", "contact", "Jane", "Smith", "at", "jane.smith@work.net"],
    ["The", "user", "is", "anonymous"],
    ["Send", "the", "report", "to", "test.user@company.org"],
]

# These are the corresponding labels for each word in the texts above.
# Notice how "John Doe" and the emails are tagged with B-PII and I-PII.
labels = [
    ["O", "O", "O", "B-PII", "I-PII", "O", "O", "O", "O", "B-PII"],
    ["O", "O", "B-PII", "I-PII", "O", "B-PII"],
    ["O", "O", "O", "O"],
    ["O", "O", "O", "O", "B-PII"],
]

# --- 4. Load the Tokenizer ---
# The tokenizer converts text into a format the model can understand (input IDs, attention masks).
# It also helps align the labels with the tokens created by the model.
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# --- 5. Tokenize and Align Labels ---
# The tokenizer may break a single word into multiple sub-words (tokens).
# We need to ensure that the labels are correctly aligned with these new tokens.
print("Tokenizing and aligning labels...")
train_encodings = tokenizer(texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else label_map[labels[word_id]]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = label_map[labels[word_id]]
            # If B-PII, convert to I-PII
            if label == label_map["B-PII"]:
                new_labels.append(label_map["I-PII"])
            else:
                new_labels.append(label)
    return new_labels


aligned_labels = []
for i, text_labels in enumerate(labels):
    word_ids = train_encodings.word_ids(i)
    aligned_labels.append(align_labels_with_tokens(text_labels, word_ids))

# Remove unnecessary columns and add the aligned labels to the encodings.
train_encodings.pop("offset_mapping")
train_encodings["labels"] = aligned_labels

# --- 6. Create a PyTorch Dataset ---
class NERDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        # Create a dictionary of tensors for the given index.
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings.labels)

train_dataset = NERDataset(train_encodings)

# --- 7. Load the Pre-trained Model ---
print("Loading pre-trained model...")
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels_list),
    id2label={i: label for i, label in enumerate(labels_list)},
    label2id=label_map
)

# --- 8. Define Training Arguments ---
# These arguments control the training process.
# In a real project, you would tune these hyperparameters.
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,          # Directory to save the model
    num_train_epochs=3,              # Total number of training epochs
    per_device_train_batch_size=8,   # Batch size per device during training
    warmup_steps=500,                # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # Strength of weight decay
    logging_dir=LOGGING_DIR,         # Directory for storing logs
    logging_steps=10,
)

# --- 9. Initialize the Trainer ---
# The Trainer class from Hugging Face simplifies the training process.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    # You can also add an evaluation dataset here.
)

# --- 10. Start Training ---
print("Starting model training...")
trainer.train()
print("Training finished!")

# --- 11. Save the Model and Tokenizer ---
# After training, save the fine-tuned model and tokenizer to the output directory.
# This allows you to load it later for inference.
print(f"Saving model and tokenizer to {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n--- Training Complete ---")
print(f"You can now find your fine-tuned model in the '{OUTPUT_DIR}' directory.")
print("You can integrate it into your application by loading it from this directory.")
