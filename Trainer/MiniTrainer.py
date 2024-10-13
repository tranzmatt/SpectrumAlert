import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
import joblib
import os
import csv

# Function to handle loading CSV data into features
def load_data_from_csv(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found.")

    features = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip the header row
        for row in reader:
            features.append([float(value) for value in row[1:]])  # Extracting feature data

    return np.array(features)

# Lite version of the RF fingerprinting model
def train_rf_fingerprinting_model(features):
    # Ensure sufficient data for training
    if len(features) < 2:
        print("Not enough data to train the model.")
        return None, None

    # Generate labels for the entire dataset (not just for the training set)
    labels = [f"Device_{i % 5}" for i in range(len(features))]  # Simulating multiple devices

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Count samples per class for cross-validation
    class_counts = Counter(y_train)
    min_samples_per_class = min(class_counts.values())
    max_cv_splits = min(3, min_samples_per_class)  # Simplified cross-validation for efficiency

    print(f"Using {max_cv_splits}-fold cross-validation (lite version).")

    # Simple RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=50,  # Reduced number of trees for faster training
        max_depth=10,     # Reduced depth for lower complexity
        random_state=42
    )

    # Train the model
    print("Training the RF fingerprinting model (lite version)...")
    model.fit(X_train, y_train)

    # Evaluate the model on test data
    y_pred = model.predict(X_test)
    print(f"Classification accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Cross-validation for performance evaluation
    skf = StratifiedKFold(n_splits=max_cv_splits)
    cv_scores = cross_val_score(model, features, labels, cv=skf)  # Use the full dataset and labels
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean cross-validation score: {np.mean(cv_scores) * 100:.2f}%")

    # Train a simple IsolationForest model for anomaly detection
    print("Training the IsolationForest model for anomaly detection (lite version)...")
    anomaly_detector = IsolationForest(contamination=0.05, random_state=42)
    anomaly_detector.fit(features)
    print("Anomaly detection model trained successfully.")

    return model, anomaly_detector

# Function to save the trained models to files
def save_model_to_file(model, filename='rf_fingerprinting_model_lite.pkl'):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def save_anomaly_model_to_file(model, filename='anomaly_detection_model_lite.pkl'):
    joblib.dump(model, filename)
    print(f"Anomaly detection model saved to {filename}")

# Main execution
if __name__ == "__main__":
    # Load the collected data from the CSV file
    data_file = 'collected_data_lite.csv'  # Lite version data
    print(f"Loading data from {data_file}...")

    try:
        features = load_data_from_csv(data_file)
        print(f"Sample features (first 5): {features[:5]}")  # Debugging statement
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)

    # Train the RF fingerprinting and anomaly detection models
    model, anomaly_model = train_rf_fingerprinting_model(features)

    # Save the trained models to files for future use
    if model is not None:
        save_model_to_file(model, 'rf_fingerprinting_model_lite.pkl')
    if anomaly_model is not None:
        save_anomaly_model_to_file(anomaly_model, 'anomaly_detection_model_lite.pkl')
