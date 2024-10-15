import numpy as np
import configparser
from rtlsdr import RtlSdr
import os
import csv
import time
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

# Lite version parameters
LITE_SAMPLE_SIZE = 128 * 1024  # Reduced sample size for Raspberry Pi
LITE_SAMPLE_RATE = 1.024e6     # Reduced sample rate for efficiency
LITE_RUNS_PER_FREQ = 3         # Fewer runs per frequency to save resources
#LITE_GAIN = 20                 # Simplified fixed gain for the lite version

# Function to read configuration file
def read_config(config_file='config.ini'):
    config = configparser.ConfigParser()

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file '{config_file}' not found.")
    
    config.read(config_file)

    if 'HAM_BANDS' not in config:
        raise ValueError("'HAM_BANDS' section missing in the config file.")

    ham_bands_str = config['HAM_BANDS'].get('bands', None)
    if ham_bands_str is None:
        raise ValueError("Missing 'bands' entry in 'HAM_BANDS' section.")
    
    ham_bands = []
    for band in ham_bands_str.split(','):
        try:
            start, end = band.split('-')
            ham_bands.append((float(start), float(end)))
        except ValueError:
            raise ValueError(f"Invalid frequency range format: {band}. Expected 'start-end'.")

    freq_step = float(config['GENERAL'].get('freq_step', 500e3))
    sample_rate = float(config['GENERAL'].get('sample_rate', LITE_SAMPLE_RATE))  # Default to lite sample rate
    runs_per_freq = int(config['GENERAL'].get('runs_per_freq', LITE_RUNS_PER_FREQ))

    return ham_bands, freq_step, sample_rate, runs_per_freq

# Lite version of feature extraction with only necessary features
def extract_features(iq_data):
    I = np.real(iq_data)
    Q = np.imag(iq_data)
    amplitude = np.sqrt(I**2 + Q**2)

    # Basic features: amplitude
    mean_amplitude = np.mean(amplitude)
    std_amplitude = np.std(amplitude)

    # Prevent negative std deviation (could happen with numerical issues)
    std_amplitude = abs(std_amplitude)

    # Return only basic features
    return [
        mean_amplitude, std_amplitude
    ]

# Function to save the collected data as a CSV
def save_data_to_csv(data, filename, header_written):
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)
    
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not header_written:
            writer.writerow(['Frequency', 'Mean_Amplitude', 'Std_Amplitude'])
        writer.writerow(data)
    
    print(f"Data saved to {filename}")

# Function to gather IQ data and process with reduced features
def gather_data_lite(sdr, ham_bands, freq_step, runs_per_freq, filename, duration_minutes):
    header_written = False
    start_time = time.time()
    duration_seconds = duration_minutes * 60  # Convert minutes to seconds

    # Initialize IsolationForest for anomaly detection
    anomaly_detector = IsolationForest(contamination=0.05, random_state=42)

    # Fit PCA on some initial data
    pca_training_data = []
    for band_start, band_end in ham_bands:
        sdr.center_freq = band_start
        iq_samples = sdr.read_samples(LITE_SAMPLE_SIZE)
        features = extract_features(iq_samples)
        pca_training_data.append(features)

    pca = PCA(n_components=min(2, len(pca_training_data[0]), len(pca_training_data)))  # Only 2 features
    pca.fit(pca_training_data)

    while time.time() - start_time < duration_seconds:
        for band_start, band_end in ham_bands:
            current_freq = band_start
            while current_freq <= band_end:
                run_features = []
                for _ in range(runs_per_freq):
                    sdr.center_freq = current_freq
                   # sdr.gain = LITE_GAIN  # Use a simplified fixed gain for the lite version
                    iq_samples = sdr.read_samples(LITE_SAMPLE_SIZE)  # Reduced sample size for efficiency
                    features = extract_features(iq_samples)
                    run_features.append(features)

                # Average features over runs
                avg_features = np.mean(run_features, axis=0)

                # Apply PCA for dimensionality reduction
                reduced_features = pca.transform([avg_features])
                data = [current_freq] + reduced_features[0].tolist()

                # Save to CSV
                save_data_to_csv(data, filename, header_written)
                header_written = True

                current_freq += freq_step

# Main execution
if __name__ == "__main__":
    # Read configuration
    ham_bands, freq_step, sample_rate, runs_per_freq = read_config('Trainer/config.ini')

    # Initialize the SDR device
    sdr = RtlSdr()
    sdr.sample_rate = sample_rate  # Set sample rate from config file
    #sdr.gain = LITE_GAIN  # Set initial gain

    # Get the duration for data gathering from user input
    duration = input("Enter the duration for data gathering (in minutes): ")
    duration = float(duration)

    # Start data gathering
    gather_data_lite(sdr, ham_bands, freq_step, runs_per_freq, 'collected_data_lite.csv', duration)

    # Close SDR device when done
    sdr.close()
    print("Closed SDR device.")
