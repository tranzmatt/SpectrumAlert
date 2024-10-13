#!/usr/bin/env python3
import os
import shutil

ASCII_LOGO = """
▗▄▄▖▗▄▄▖  ▗▄▄▄▖▗▄▄▖ ▗▄▄▄▖▗▄▄▖ ▗▖ ▗▖▗▖  ▗▖     ▗▄▖ ▗▖   ▗▄▄▄▖▗▄▄▖▗▄▄▄▖
▐▌   ▐▌ ▐▌▐▌   ▐▌     █  ▐▌ ▐▌▐▌ ▐▌▐▛▚▞▜▌    ▐▌ ▐▌▐▌   ▐▌   ▐▌ ▐▌ █  
 ▝▀▚▖▐▛▀▘ ▐▛▀▘▐▌     █  ▐▛▀▚▖▐▌ ▐▌▐▌  ▐▌    ▐▛▀▜▌▐▌   ▐▛▀▘▐▛▀▚▖ █  
▗▄▄▞▘▐▌   ▐▙▄▄▖▝▚▄▄▖  █  ▐▌ ▐▌▝▚▄▞▘▐▌  ▐▌    ▐▌ ▐▌▐▙▄▄▖▐▙▄▄▖▐▌ ▐▌ █                                                            
                                                                  
"""

# File paths for lite and normal versions
DATA_FILE = "collected_data_lite.csv"
NORMAL_DATA_FILE = "collected_iq_data.csv"
MODEL_FILE = "rf_fingerprinting_model_lite.pkl"
NORMAL_MODEL_FILE = "rf_fingerprinting_model.pkl"
ANOMALY_MODEL_FILE = "anomaly_detection_model_lite.pkl"
NORMAL_ANOMALY_MODEL_FILE = "anomaly_detection_model.pkl"

def start_from_scratch():
    """Delete existing datasets and models."""
    # Delete Lite files
    for file in [DATA_FILE, MODEL_FILE, ANOMALY_MODEL_FILE]:
        if os.path.exists(file):
            os.remove(file)
            print(f"Deleted {file}")

    # Delete Normal files
    for file in [NORMAL_DATA_FILE, NORMAL_MODEL_FILE, NORMAL_ANOMALY_MODEL_FILE]:
        if os.path.exists(file):
            os.remove(file)
            print(f"Deleted {file}")

    print("Starting from scratch. All datasets and models have been deleted.")

def automate_process(duration):
    """Automate data gathering, model training, and monitoring."""
    # Check for lite dataset and models
    if os.path.exists(DATA_FILE):
        print(f"Lite dataset found: {DATA_FILE}. Training the lite model...")
        os.system("python3 Trainer/MiniTrainer.py")
    elif os.path.exists(MODEL_FILE) and os.path.exists(ANOMALY_MODEL_FILE):
        print(f"Lite models found: {MODEL_FILE} and {ANOMALY_MODEL_FILE}. Starting lite monitor...")
        os.system("python3 Trainer/MiniMonitor.py")
    # Check for normal dataset and models
    elif os.path.exists(NORMAL_DATA_FILE):
        print(f"Normal dataset found: {NORMAL_DATA_FILE}. Training the normal model...")
        os.system("python3 Trainer/ModelTrainer.py")
    elif os.path.exists(NORMAL_MODEL_FILE) and os.path.exists(NORMAL_ANOMALY_MODEL_FILE):
        print(f"Normal models found: {NORMAL_MODEL_FILE} and {NORMAL_ANOMALY_MODEL_FILE}. Starting normal monitor...")
        os.system("python3 Trainer/Monitor.py")
    else:
        print("No dataset or models found. Starting data gathering...")
        version_choice = input("Do you want to run the lite version for Raspberry Pi? (y/n): ").lower()
        if version_choice == 'y':
            script = "Trainer/MiniData.py"
        else:
            script = "Trainer/DataGathering.py"
        
        # Use the duration already provided
        os.system(f"python3 {script} {duration}")
        
        # Check if gathering lite or normal data and train accordingly
        if version_choice == 'y':
            print("Lite data gathering completed. Training the lite model...")
            os.system("python3 Trainer/MiniTrainer.py")
            print("Lite model training completed. Starting lite monitor...")
            os.system("python3 Trainer/MiniMonitor.py")
        else:
            print("Normal data gathering completed. Training the normal model...")
            os.system("python3 Trainer/ModelTrainer.py")
            print("Normal model training completed. Starting normal monitor...")
            os.system("python3 Trainer/Monitor.py")

def main():
    while True:
        print(ASCII_LOGO)
        print("Welcome to Spectrum Alert")
        print("Please choose an option:")
        print("1. Gather Data (DataGathering.py or MiniData.py)")
        print("2. Train Model (ModelTrainer.py or MiniTrainer.py)")
        print("3. Monitor Spectrum (Monitor.py or MiniMonitor.py)")
        print("4. Automate: Gather Data -> Train Model -> Monitor Spectrum")
        print("5. Automate: Train model or run monitor depending on existing data/models")
        print("6. Start from scratch (delete datasets and models)")
        print("7. Exit")

        choice = input("Enter your choice (1-7): ")

        if choice == "1":
            version_choice = input("Do you want to run the lite version for Raspberry Pi? (y/n): ").lower()
            if version_choice == 'y':
                script = "Trainer/MiniData.py"
            else:
                script = "Trainer/DataGathering.py"
            
            duration = input("Enter the duration for data gathering (in minutes): ")
            os.system(f"python3 {script} {duration}")

        elif choice == "2":
            version_choice = input("Do you want to run the lite version for Raspberry Pi? (y/n): ").lower()
            if version_choice == 'y':
                os.system("python3 Trainer/MiniTrainer.py")
            else:
                os.system("python3 Trainer/ModelTrainer.py")

        elif choice == "3":
            version_choice = input("Do you want to run the lite version for Raspberry Pi? (y/n): ").lower()
            if version_choice == 'y':
                os.system("python3 Trainer/MiniMonitor.py")
            else:
                os.system("python3 Trainer/Monitor.py")

        elif choice == "4":
            # Get the duration before automating the process
            duration = input("Enter the duration for data gathering (in minutes): ")
            print("Automating process: Gather Data -> Train Model -> Monitor Spectrum")
            automate_process(duration)

        elif choice == "5":
            # For option 5, no need for duration since it's only used if data gathering starts
            print("Checking for existing dataset or model...")
            automate_process(None)

        elif choice == "6":
            print("Starting from scratch...")
            start_from_scratch()

        elif choice == "7":
            print("Exiting... Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.\n")

if __name__ == "__main__":
    main()
