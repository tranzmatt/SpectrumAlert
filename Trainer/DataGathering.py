import numpy as np
import configparser
from rtlsdr import RtlSdr
import os
import json

# Function to read and parse the config file
def read_config(config_file='config.ini'):
    config = configparser.ConfigParser()
    config.read(config_file)

    # Parse HAM bands
    ham_bands_str = config['HAM_BANDS']['bands']
    ham_bands = []
    for band in ham_bands_str.split(','):
        start, end = band.split('-')
        ham_bands.append((float(start), float(end)))

    # Parse general settings
    freq_step = float(config['GENERAL']['freq_step'])
    sample_rate = float(config['GENERAL']['sample_rate'])
    runs_per_freq = int(config['GENERAL']['runs_per_freq'])

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
def gather_iq_data(sdr, ham_bands, freq_step, runs_per_freq):
    data = []

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
            data.append({
                'frequency': current_freq,
                'features': avg_features.tolist()
            })

            print(f"Collected data at {current_freq / 1e6:.2f} MHz")

            # Move to the next frequency step
            current_freq += freq_step

    return data

# Function to save the collected data to a file
def save_data_to_file(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(data, f)
    print(f"Data saved to {filename}")

# Main execution
if __name__ == "__main__":
    try:
        # Read the configuration file
        ham_bands, freq_step, sample_rate, runs_per_freq = read_config()

        # Instantiate RTL-SDR
        sdr = RtlSdr()
        sdr.sample_rate = sample_rate
        sdr.gain = 'auto'  # Automatic gain control

        # Collect IQ data and extract features
        print("Starting IQ data collection...")
        collected_data = gather_iq_data(sdr, ham_bands, freq_step, runs_per_freq)

        # Save the collected data to a file
        save_data_to_file(collected_data, 'collected_iq_data.json')

    except KeyboardInterrupt:
        print("Data collection interrupted by user.")
    finally:
        sdr.close()
        print("Closed SDR device.")
