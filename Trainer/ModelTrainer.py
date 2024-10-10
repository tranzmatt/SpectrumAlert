import json
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib  # Used to save the model

# Function to load the collected IQ data from the JSON file
def load_data_from_file(filename):
    with open(filename, 'r') as f:
        data = json.load(f)

    # Extract features and corresponding frequencies
    features = np.array([entry['features'] for entry in data])
    frequencies = np.array([entry['frequency'] for entry in data])

    return features, frequencies

# Function to train the Isolation Forest anomaly detection model
def train_anomaly_model(features):
    # Splitting data into training and testing sets
    X_train, X_test = train_test_split(features, test_size=0.2, random_state=42)

    # Initializing the IsolationForest model
    model = IsolationForest(contamination=0.01, random_state=42)

    # Train the model on the training data
    print("Training the Isolation Forest model...")
    model.fit(X_train)

    # Use the trained model to predict anomalies on the test set
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Since IsolationForest returns -1 for anomalies and 1 for normal data, let's convert to binary
    y_pred_train = np.where(y_pred_train == 1, 0, 1)  # 0: normal, 1: anomaly
    y_pred_test = np.where(y_pred_test == 1, 0, 1)

    return model, X_train, X_test, y_pred_train, y_pred_test

# Function to save the trained model to a file
def save_model_to_file(model, filename='anomaly_detection_model.pkl'):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

# Function to evaluate model performance
def evaluate_model(X_train, X_test, y_pred_train, y_pred_test):
    print("Training set evaluation:")
    print(f"Train accuracy: {accuracy_score(np.zeros(len(X_train)), y_pred_train) * 100:.2f}%")
    print("Testing set evaluation:")
    print(f"Test accuracy: {accuracy_score(np.zeros(len(X_test)), y_pred_test) * 100:.2f}%")
    print("\nClassification report (on test set):")
    print(classification_report(np.zeros(len(X_test)), y_pred_test, target_names=["Normal", "Anomaly"]))

# Main execution
if __name__ == "__main__":
    # Load the collected data from the JSON file
    data_file = 'collected_iq_data.json'
    print(f"Loading data from {data_file}...")
    features, frequencies = load_data_from_file(data_file)

    # Train the anomaly detection model
    model, X_train, X_test, y_pred_train, y_pred_test = train_anomaly_model(features)

    # Evaluate the model
    evaluate_model(X_train, X_test, y_pred_train, y_pred_test)

    # Save the trained model to a file for future use
    save_model_to_file(model)
