
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, accuracy_score
import joblib

# --- 1. Configuration ---
DATA_FILE = "risk_data.csv"
CLASSIFIER_OUTPUT_FILE = "risk_classifier.joblib"
ANOMALY_DETECTOR_OUTPUT_FILE = "anomaly_detector.joblib"

# --- 2. Load Dataset ---
print(f"Loading data from {DATA_FILE}...")
df = pd.read_csv(DATA_FILE)

# Define features and target
features = ["pii_count", "custom_rules_matches", "high_entropy_strings_count"]
target = "risk_level"

X = df[features]
y = df[target]

# --- 3. Train Risk Scoring (Classification) Model ---
print("\n--- Training Classification Model ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

classifier = LogisticRegression(random_state=42, class_weight='balanced')
classifier.fit(X_train, y_train)

# Evaluate the classifier
y_pred = classifier.predict(X_test)
print(f"Classifier Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))

# Save the trained classifier
print(f"Saving classifier to {CLASSIFIER_OUTPUT_FILE}...")
joblib.dump(classifier, CLASSIFIER_OUTPUT_FILE)

# --- 4. Train Anomaly Detection Model ---
print("\n--- Training Anomaly Detection Model ---")
# We train the anomaly detector on all data to learn the distribution of what's "normal"
anomaly_detector = IsolationForest(contamination='auto', random_state=42)
anomaly_detector.fit(X)

# Save the trained anomaly detector
print(f"Saving anomaly detector to {ANOMALY_DETECTOR_OUTPUT_FILE}...")
joblib.dump(anomaly_detector, ANOMALY_DETECTOR_OUTPUT_FILE)

print("\n--- Model Training Complete ---")
print(f"Models saved as '{CLASSIFIER_OUTPUT_FILE}' and '{ANOMALY_DETECTOR_OUTPUT_FILE}'.")
