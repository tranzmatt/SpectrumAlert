import numpy as np
import configparser
from rtlsdr import RtlSdr
import os
import csv
import time

# Function to read and parse the config file
def read_config(config_file='Trainer/config.ini'):
    config = configparser.ConfigParser()

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file '{config_file}' not found.")
    
    config.read(config_file)

    # Check if 'HAM_BANDS' section exists in config
    if 'HAM_BANDS' not in config:
        raise ValueError("'HAM_BANDS' section missing in the config file.")

    # Parse HAM bands
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

    # Parse general settings
    freq_step = float(config['GENERAL'].get('freq_step', 500e3))
    sample_rate = float(config['GENERAL'].get('sample_rate', 2.048e6))
    runs_per_freq = int(config['GENERAL'].get('runs_per_freq', 5))

    return ham_bands, freq_step, sample_rate, runs_per_freq

# Function to extract features from IQ data
def extract_features(iq_data):
    I = np.real(iq_data)
    Q = np.imag(iq_data)
    amplitude = np.sqrt(I**2 + Q**2)
    fft_values = np.fft.fft(iq_data)
    fft_magnitude = np.abs(fft_values)
    return [np.mean(amplitude), np.std(amplitude), np.mean(fft_magnitude), np.std(fft_magnitude)]

# Function to gather IQ data and extract features
def gather_iq_data_continuous(sdr, ham_bands, freq_step, runs_per_freq, filename):
    header_written = False  # To ensure we write the CSV header only once
    start_time = time.time()

    while True:  # Infinite loop to gather data continuously
        for band_start, band_end in ham_bands:
            current_freq = band_start
            while current_freq <= band_end:
                run_features = []
                for _ in range(runs_per_freq):
                    sdr.center_freq = current_freq
                    iq_samples = sdr.read_samples(128 * 1024)  # Read samples
                    features = extract_features(iq_samples)
                    run_features.append(features)

                # Average over multiple runs
                avg_features = np.mean(run_features, axis=0)
                data = [current_freq] + avg_features.tolist()

                print(f"Collected data at {current_freq / 1e6:.2f} MHz")

                # Save the collected data to CSV
                save_data_to_csv(data, filename, header_written)
                header_written = True  # Header has been written after the first save

                # Move to the next frequency step
                current_freq += freq_step

        # Reset timer to save every 60 seconds
        current_time = time.time()
        if current_time - start_time > 60:
            start_time = current_time  # Reset timer

# Function to save the collected data as a CSV
def save_data_to_csv(data, filename, header_written):
    directory = os.path.dirname(filename)

    # Create directory if it doesn't exist and if a directory path is provided
    if directory:
        os.makedirs(directory, exist_ok=True)
    
    # Save data to CSV file
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not header_written:
            writer.writerow(['Frequency', 'Mean_Amplitude', 'Std_Amplitude', 'Mean_FFT_Magnitude', 'Std_FFT_Magnitude'])
        writer.writerow(data)
    
    print(f"Data saved to {filename}")

# Main execution
if __name__ == "__main__":
    sdr = None  # Initialize sdr to None
    try:
        # Read the configuration file
        ham_bands, freq_step, sample_rate, runs_per_freq = read_config()

        # Instantiate RTL-SDR
        sdr = RtlSdr()
        sdr.sample_rate = sample_rate
        sdr.gain = 'auto'  # Automatic gain control

        # Start continuous IQ data collection
        print("Starting continuous IQ data collection...")
        gather_iq_data_continuous(sdr, ham_bands, freq_step, runs_per_freq, 'collected_iq_data.csv')

    except KeyboardInterrupt:
        print("Data collection interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if sdr is not None:  # Only close SDR if it was successfully initialized
            sdr.close()
            print("Closed SDR device.")