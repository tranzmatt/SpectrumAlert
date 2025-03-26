import argparse
import csv
import os
import sys
from collections import Counter

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold


# Function to handle loading CSV data into features
def load_data_from_csv(local_filename):
    if not os.path.exists(local_filename):
        raise FileNotFoundError(f"File {local_filename} not found.")

    local_features = []
    with open(local_filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip the header row
        for row in reader:
            local_features.append([float(value) for value in row[1:]])  # Extracting feature data

    return np.array(local_features)


# Lite version of the RF fingerprinting model
def train_rf_fingerprinting_model(local_features):
    # Ensure sufficient data for training
    if len(local_features) < 2:
        print("Not enough data to train the model.")
        return None, None

    # Generate labels for the entire dataset (not just for the training set)
    labels = [f"Device_{i % 5}" for i in range(len(local_features))]  # Simulating multiple devices

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(local_features, labels, test_size=0.2, random_state=42)

    # Count samples per class for cross-validation
    class_counts = Counter(y_train)
    min_samples_per_class = min(class_counts.values())
    max_cv_splits = min(3, min_samples_per_class)  # Simplified cross-validation for efficiency

    print(f"Using {max_cv_splits}-fold cross-validation (lite version).")

    # Simple RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=50,  # Reduced number of trees for faster training
        max_depth=10,  # Reduced depth for lower complexity
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


if __name__ == "__main__":
    try:
        # ✅ Use argparse for command-line parsing
        parser = argparse.ArgumentParser(description="Spectrum Monitoring with SDR.")
        parser.add_argument("-i", "--input", type=str, default="collected_data_lite.csv",
                            help="Path to the collected lite data file (default: collected_data_lite.csv)")
        parser.add_argument("-a", "--anomaly", type=str, default="anomaly_detection_model_lite.pkl",
                            help="Path to the output lite anomaly file (default: anomaly_detection_model_lite.pkl)")
        parser.add_argument("-f", "--fingerprint", type=str, default="rf_fingerprinting_model_lite.pkl",
                            help="Path to the output lite fingerprint file (default: rf_fingerprinting_model_lite.pkl)")

        args = parser.parse_args()

        input_file = args.input
        fingerprint_model_file = args.fingerprint
        anomaly_detection_model_file = args.anomaly

        print(f"Loading data from {input_file}...")

        try:
            features = load_data_from_csv(input_file)
            print(f"Sample features (first 5): {features[:5]}")  # Debugging statement
        except Exception as e:
            print(f"Error loading data: {e}")
            exit(1)

        # Train the RF fingerprinting and anomaly detection models
        fingerprint_model, anomaly_model = train_rf_fingerprinting_model(features)

        print(f"Saving to {fingerprint_model_file} / {anomaly_detection_model_file}...")

        # Save the trained models to files for future use
        if fingerprint_model is not None:
            save_model_to_file(fingerprint_model, fingerprint_model_file)
        if anomaly_model is not None:
            save_anomaly_model_to_file(anomaly_model, anomaly_detection_model_file)

    except KeyboardInterrupt:
        sys.exit(0)

    except Exception as e:
        print(f"An error occurred: {e}")
