import numpy as np
import configparser
from rtlsdr import RtlSdr
import os
import csv
import time
from sklearn.decomposition import PCA
from concurrent.futures import ThreadPoolExecutor
import sys
import threading

# Thread lock for safe file access
file_lock = threading.Lock()

# Shared object for header written status
header_lock = threading.Lock()
header_written = False

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
    amplitude = np.sqrt(I**2 + Q**2)  # Magnitude of the complex signal
    phase = np.unwrap(np.angle(iq_data))  # Unwrap the phase

    # FFT of the signal
    fft_values = np.fft.fft(iq_data)
    fft_magnitude = np.abs(fft_values)

    # Mean and standard deviation of the amplitude
    mean_amplitude = np.mean(amplitude)
    std_amplitude = np.std(amplitude)

    # Mean and standard deviation of the FFT magnitude
    mean_fft_magnitude = np.mean(fft_magnitude)
    std_fft_magnitude = np.std(fft_magnitude)

    # Skewness and kurtosis of amplitude
    if std_amplitude != 0:
        skew_amplitude = np.mean((amplitude - mean_amplitude) ** 3) / (std_amplitude ** 3)
        kurt_amplitude = np.mean((amplitude - mean_amplitude) ** 4) / (std_amplitude ** 4)
    else:
        skew_amplitude = 0
        kurt_amplitude = 0

    # Skewness and kurtosis of phase
    std_phase = np.std(phase)
    mean_phase = np.mean(phase)
    if std_phase != 0:
        skew_phase = np.mean((phase - mean_phase) ** 3) / (std_phase ** 3)
        kurt_phase = np.mean((phase - mean_phase) ** 4) / (std_phase ** 4)
    else:
        skew_phase = 0
        kurt_phase = 0

    # Cyclostationary autocorrelation (average of autocorrelation)
    if len(amplitude) > 1:
        cyclo_autocorr = np.abs(np.correlate(amplitude, amplitude, mode='full')[len(amplitude) // 2:]).mean()
    else:
        cyclo_autocorr = 0

    # Spectral entropy (FFT magnitude normalized)
    fft_magnitude_sum = np.sum(fft_magnitude)
    if fft_magnitude_sum > 0:
        normalized_fft = fft_magnitude / fft_magnitude_sum
        spectral_entropy = -np.sum(normalized_fft * np.log2(normalized_fft + 1e-12))  # Add small value to avoid log(0)
    else:
        spectral_entropy = 0

    # Peak-to-Average Power Ratio (PAPR)
    if mean_amplitude > 0:
        papr = np.max(amplitude) ** 2 / np.mean(amplitude ** 2)
    else:
        papr = 0

    # Band Energy Ratio (lower half of FFT vs total)
    fft_magnitude_half = fft_magnitude[:len(fft_magnitude) // 2]
    if fft_magnitude_sum > 0:
        band_energy_ratio = np.sum(fft_magnitude_half) / fft_magnitude_sum
    else:
        band_energy_ratio = 0

    return [
        mean_amplitude, std_amplitude, mean_fft_magnitude, std_fft_magnitude,
        skew_amplitude, kurt_amplitude, skew_phase, kurt_phase, cyclo_autocorr,
        spectral_entropy, papr, band_energy_ratio
    ]

# Function to save the collected data as a CSV
def save_data_to_csv(data, filename):
    global header_written
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with file_lock:  # Ensure thread-safe file writing
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)

            with header_lock:
                if not header_written:
                    writer.writerow(['Frequency', 'Mean_Amplitude', 'Std_Amplitude', 'Mean_FFT_Magnitude', 'Std_FFT_Magnitude',
                                     'Skew_Amplitude', 'Kurt_Amplitude', 'Skew_Phase', 'Kurt_Phase', 'Cyclo_Autocorr',
                                     'Spectral_Entropy', 'PAPR', 'Band_Energy_Ratio'])
                    header_written = True  # Update the flag after writing the header
            
            # Debug: Print the data being written to the CSV
            print(f"Writing to CSV: {data}")
            writer.writerow(data)

    print(f"Data saved to {filename}")

# Function to scan a single band
def scan_band(sdr, band_start, band_end, freq_step, runs_per_freq, filename, pca):
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
        reduced_features = pca.transform([avg_features])  # Use avg_features directly for the complete feature list
        
        data = [current_freq] + avg_features.tolist()  # Add the frequency and all features to the data
        
        # Save to CSV
        save_data_to_csv(data, filename)
        
        # Move to the next frequency
        current_freq += freq_step

# Main function for parallel processing of bands
def gather_iq_data_parallel(sdr, ham_bands, freq_step, runs_per_freq, filename, duration_minutes):
    start_time = time.time()
    duration_seconds = duration_minutes * 60

    # Collect initial data to fit PCA
    pca_training_data = []
    for band_start, band_end in ham_bands:
        sdr.center_freq = band_start
        iq_samples = sdr.read_samples(256 * 1024)
        features = extract_features(iq_samples)
        pca_training_data.append(features)

    num_features = len(pca_training_data[0])
    n_components = min(8, len(pca_training_data), num_features)

    pca = PCA(n_components=n_components)
    pca.fit(pca_training_data)

    # Parallel scanning of bands
    with ThreadPoolExecutor() as executor:
        futures = []
        for band_start, band_end in ham_bands:
            futures.append(executor.submit(scan_band, sdr, band_start, band_end, freq_step, runs_per_freq, filename, pca))

        # Wait for all threads to finish
        for future in futures:
            future.result()

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
