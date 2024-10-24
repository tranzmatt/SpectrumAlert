import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score , StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import DBSCAN
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
            features.append([float(value) for value in row[1:]])

    return np.array(features)


def train_rf_fingerprinting_model(features):
    # Splitting data into training and testing sets
    if len(features) < 2:
        print("Not enough data to train the model.")
        return None, None

    X_train, X_test = train_test_split(features, test_size=0.2, random_state=42)

    # Dynamically generate labels (simulate multiple devices)
    labels = [f"Device_{i % 10}" for i in range(len(X_train))]

    # Count samples per class
    class_counts = Counter(labels)
    min_samples_per_class = min(class_counts.values())

    # Determine maximum number of splits for cross-validation
        # Determine maximum number of splits for cross-validation
    min_samples_per_class = min(class_counts.values())
    max_cv_splits = min(5, min_samples_per_class)  # Ensure cv doesn't exceed the smallest class size
    max_cv_splits = max(2, max_cv_splits)  # Ensure at least 2 splits

    print(f"Using {max_cv_splits}-fold cross-validation (based on smallest class size).")

    # Initializing the RandomForestClassifier model
    model = RandomForestClassifier(random_state=42)

    # Setting up hyperparameter tuning with GridSearchCV and StratifiedKFold
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    skf = StratifiedKFold(n_splits=max_cv_splits)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=skf, n_jobs=-1, verbose=2)

    # Train the model on the training data
    print("Training the RF fingerprinting model with hyperparameter tuning...")
    grid_search.fit(X_train, labels)

    # Best model from grid search
    best_model = grid_search.best_estimator_
    print(f"Best parameters found: {grid_search.best_params_}")

    # Evaluate the model on the test data
    y_pred = best_model.predict(X_test)
    y_test_labels = [f"Device_{i % 10}" for i in range(len(X_test))]  # Simulated labels for evaluation
    print(f"Classification accuracy: {accuracy_score(y_test_labels, y_pred) * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test_labels, y_pred))

    # Cross-validation for more reliable performance evaluation
    cv_labels = [f"Device_{i % 10}" for i in range(len(features))]  # Consistent labels for cross-validation
    cv_scores = cross_val_score(best_model, features, cv_labels, cv=skf)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean cross-validation score: {np.mean(cv_scores) * 100:.2f}%")

    # Train an IsolationForest model for anomaly detection
    print("Training the IsolationForest model for anomaly detection...")
    anomaly_detector = IsolationForest(contamination=0.05, random_state=42)
    anomaly_detector.fit(features)
    print("Anomaly detection model trained successfully.")

    return best_model, anomaly_detector
# Function to save the trained models to files
def save_model_to_file(model, filename='rf_fingerprinting_model.pkl'):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def save_anomaly_model_to_file(model, filename='anomaly_detection_model.pkl'):
    joblib.dump(model, filename)
    print(f"Anomaly detection model saved to {filename}")

# Main execution
if __name__ == "__main__":
    # Load the collected data from the CSV file
    data_file = 'collected_iq_data.csv'
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
        save_model_to_file(model, 'rf_fingerprinting_model.pkl')
    if anomaly_model is not None:
        save_anomaly_model_to_file(anomaly_model, 'anomaly_detection_model.pkl')