import numpy as np
import configparser
from rtlsdr import RtlSdr
import os
import csv
import time
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from concurrent.futures import ThreadPoolExecutor
import sys
import threading

# Thread lock for safe file access
file_lock = threading.Lock()

# Function to read and parse the config file
def read_config(config_file='Trainer/config.ini'):
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
    sample_rate = float(config['GENERAL'].get('sample_rate', 2.048e6))
    runs_per_freq = int(config['GENERAL'].get('runs_per_freq', 5))

    return ham_bands, freq_step, sample_rate, runs_per_freq

# Function to extract enhanced features from IQ data
def extract_features(iq_data):
    I = np.real(iq_data)
    Q = np.imag(iq_data)
    amplitude = np.sqrt(I**2 + Q**2)
    phase = np.unwrap(np.angle(iq_data))

    fft_values = np.fft.fft(iq_data)
    fft_magnitude = np.abs(fft_values)
    mean_amplitude = np.mean(amplitude)
    std_amplitude = np.std(amplitude)
    mean_fft_magnitude = np.mean(fft_magnitude)
    std_fft_magnitude = np.std(fft_magnitude)

    skew_amplitude = np.mean((amplitude - mean_amplitude) ** 3) / (std_amplitude ** 3)
    kurt_amplitude = np.mean((amplitude - mean_amplitude) ** 4) / (std_amplitude ** 4)
    skew_phase = np.mean((phase - np.mean(phase)) ** 3) / (np.std(phase) ** 3)
    kurt_phase = np.mean((phase - np.mean(phase)) ** 4) / (np.std(phase) ** 4)

    cyclo_autocorr = np.abs(np.correlate(amplitude, amplitude, mode='full')[len(amplitude)//2:]).mean()
    spectral_entropy = -np.sum((fft_magnitude / np.sum(fft_magnitude)) * np.log2(fft_magnitude / np.sum(fft_magnitude) + 1e-12))
    papr = np.max(amplitude) ** 2 / np.mean(amplitude ** 2)
    band_energy_ratio = np.sum(fft_magnitude[:len(fft_magnitude)//2]) / np.sum(fft_magnitude)

    return [
        mean_amplitude, std_amplitude, mean_fft_magnitude, std_fft_magnitude,
        skew_amplitude, kurt_amplitude, skew_phase, kurt_phase, cyclo_autocorr,
        spectral_entropy, papr, band_energy_ratio
    ]

# Function to save the collected data as a CSV
def save_data_to_csv(data, filename, header_written):
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with file_lock:  # Ensure thread-safe file writing
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            if not header_written:
                writer.writerow(['Frequency', 'Mean_Amplitude', 'Std_Amplitude', 'Mean_FFT_Magnitude', 'Std_FFT_Magnitude',
                                 'Skew_Amplitude', 'Kurt_Amplitude', 'Skew_Phase', 'Kurt_Phase', 'Cyclo_Autocorr',
                                 'Spectral_Entropy', 'PAPR', 'Band_Energy_Ratio'])
                header_written = True  # Update the flag after writing the header
            writer.writerow(data)

    print(f"Data saved to {filename}")
    return header_written  # Return the updated flag

def scan_band(sdr, band_start, band_end, freq_step, runs_per_freq, filename, anomaly_detector, pca, header_written):
    current_freq = band_start
    collected_features = []
    while current_freq <= band_end:
        run_features = []
        for _ in range(runs_per_freq):
            sdr.center_freq = current_freq
            iq_samples = sdr.read_samples(256 * 1024)
            features = extract_features(iq_samples)
            run_features.append(features)

        avg_features = np.mean(run_features, axis=0)
        
        # Apply PCA for dimensionality reduction
        reduced_features = pca.transform([avg_features])
        data = [current_freq] + reduced_features[0].tolist()
        
        collected_features.append(avg_features)
        header_written = save_data_to_csv(data, filename, header_written)  # Update header_written after first save
        
        # Anomaly detection
        if len(collected_features) > 50:
            anomaly_detector.fit(collected_features)
            is_anomaly = anomaly_detector.predict([avg_features])[0] == -1
            if is_anomaly:
                print(f"Anomaly detected at {current_freq / 1e6:.2f} MHz")
        
        current_freq += freq_step

    return header_written  # Return updated flag

# Main function for parallel processing of bands
def gather_iq_data_parallel(sdr, ham_bands, freq_step, runs_per_freq, filename, duration_minutes):
    header_written = False  # Initial flag value
    start_time = time.time()
    duration_seconds = duration_minutes * 60
    
    # Initialize IsolationForest
    anomaly_detector = IsolationForest(contamination=0.05, random_state=42)
    
    # Collect initial data to fit PCA
    pca_training_data = []
    for band_start, band_end in ham_bands:
        sdr.center_freq = band_start
        iq_samples = sdr.read_samples(256 * 1024)
        features = extract_features(iq_samples)
        pca_training_data.append(features)

    # Determine dynamic number of components for PCA
    num_features = len(pca_training_data[0])
    num_samples = len(pca_training_data)
    n_components = min(8, num_samples, num_features)

    # Fit PCA on the collected data
    pca = PCA(n_components=n_components)
    pca.fit(pca_training_data)

    # Parallel scanning of bands
    with ThreadPoolExecutor() as executor:
        futures = []
        for band_start, band_end in ham_bands:
            futures.append(executor.submit(scan_band, sdr, band_start, band_end, freq_step, runs_per_freq, filename, anomaly_detector, pca, header_written))

        # Retrieve results and update header_written
        for future in futures:
            header_written = future.result()  # Update header_written after each band scan

# Main execution
if __name__ == "__main__":
    sdr = None
    try:
        if len(sys.argv) > 1:
            duration = float(sys.argv[1])
        else:
            raise ValueError("No duration specified. Please provide the duration in minutes as an argument.")

        ham_bands, freq_step, sample_rate, runs_per_freq = read_config()
        sdr = RtlSdr()
        sdr.sample_rate = sample_rate

        print(f"Starting IQ data collection for {duration} minutes...")
        gather_iq_data_parallel(sdr, ham_bands, freq_step, runs_per_freq, 'collected_iq_data.csv', duration)

    except KeyboardInterrupt:
        print("Data collection interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if sdr is not None:
            sdr.close()
            print("Closed SDR device.")
