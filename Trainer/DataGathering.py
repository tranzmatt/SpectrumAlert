import numpy as np
import configparser
from rtlsdr import RtlSdr
import os
import csv
import time
from sklearn.ensemble import IsolationForest
import sys

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

# Function to extract enhanced features from IQ data for RF fingerprinting
def extract_features(iq_data):
    I = np.real(iq_data)
    Q = np.imag(iq_data)
    amplitude = np.sqrt(I**2 + Q**2)
    phase = np.unwrap(np.angle(iq_data))

    # Basic features
    fft_values = np.fft.fft(iq_data)
    fft_magnitude = np.abs(fft_values)
    mean_amplitude = np.mean(amplitude)
    std_amplitude = np.std(amplitude)
    mean_fft_magnitude = np.mean(fft_magnitude)
    std_fft_magnitude = np.std(fft_magnitude)

    # Higher-order statistics for RF fingerprinting
    skew_amplitude = np.mean((amplitude - mean_amplitude) ** 3) / (std_amplitude ** 3)
    kurt_amplitude = np.mean((amplitude - mean_amplitude) ** 4) / (std_amplitude ** 4)
    skew_phase = np.mean((phase - np.mean(phase)) ** 3) / (np.std(phase) ** 3)
    kurt_phase = np.mean((phase - np.mean(phase)) ** 4) / (np.std(phase) ** 4)

    # Cyclostationary features (simplified)
    cyclo_autocorr = np.abs(np.correlate(amplitude, amplitude, mode='full')[len(amplitude)//2:]).mean()

    # Additional features for enrichment
    spectral_entropy = -np.sum((fft_magnitude / np.sum(fft_magnitude)) * np.log2(fft_magnitude / np.sum(fft_magnitude) + 1e-12))
    papr = np.max(amplitude) ** 2 / np.mean(amplitude ** 2)
    band_energy_ratio = np.sum(fft_magnitude[:len(fft_magnitude)//2]) / np.sum(fft_magnitude)

    return [
        mean_amplitude, std_amplitude, mean_fft_magnitude, std_fft_magnitude,
        skew_amplitude, kurt_amplitude, skew_phase, kurt_phase, cyclo_autocorr,
        spectral_entropy, papr, band_energy_ratio
    ]


# Function to gather IQ data with a time limit
def gather_iq_data_continuous(sdr, ham_bands, freq_step, runs_per_freq, filename, duration_minutes):
    header_written = False
    start_time = time.time()
    duration_seconds = duration_minutes * 60  # Convert minutes to seconds
    collected_features = []

    # Initialize IsolationForest for anomaly detection
    anomaly_detector = IsolationForest(contamination=0.05, random_state=42)

    while time.time() - start_time < duration_seconds:  # Run for the specified duration
        for band_start, band_end in ham_bands:
            current_freq = band_start
            while current_freq <= band_end:
                run_features = []
                for _ in range(runs_per_freq):
                    sdr.center_freq = current_freq
                    sdr.gain = adjust_gain(sdr)
                    iq_samples = sdr.read_samples(256 * 1024)
                    features = extract_features(iq_samples)
                    run_features.append(features)

                avg_features = np.mean(run_features, axis=0)
                data = [current_freq] + avg_features.tolist()
                collected_features.append(avg_features)

                print(f"Collected data at {current_freq / 1e6:.2f} MHz")

                if len(collected_features) > 50:
                    anomaly_detector.fit(collected_features)
                    is_anomaly = anomaly_detector.predict([avg_features])[0] == -1
                    if is_anomaly:
                        print(f"Anomaly detected at {current_freq / 1e6:.2f} MHz")

                save_data_to_csv(data, filename, header_written)
                header_written = True

                current_freq += freq_step

# Function to dynamically adjust the SDR gain based on signal strength
def adjust_gain(sdr):
    # Adjust gain dynamically by evaluating signal power and choosing an optimal gain value
    power = np.mean(np.abs(sdr.read_samples(1024)) ** 2)
    if power < -40:
        return 40  # Set to high gain if the signal is weak
    elif power < -20:
        return 20  # Set to moderate gain for medium strength signals
    else:
        return 10  # Set to low gain for strong signals

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
            writer.writerow(['Frequency', 'Mean_Amplitude', 'Std_Amplitude', 'Mean_FFT_Magnitude', 'Std_FFT_Magnitude',
                             'Skew_Amplitude', 'Kurt_Amplitude', 'Skew_Phase', 'Kurt_Phase', 'Cyclo_Autocorr',
                             'Spectral_Entropy', 'PAPR', 'Band_Energy_Ratio'])
        writer.writerow(data)
    
    print(f"Data saved to {filename}")

## Main execution
if __name__ == "__main__":
    sdr = None
    try:
        # Check if duration was passed as argument
        if len(sys.argv) > 1:
            duration = float(sys.argv[1])  # Get the duration from the command-line argument
        else:
            raise ValueError("No duration specified. Please provide the duration in minutes as an argument.")

        ham_bands, freq_step, sample_rate, runs_per_freq = read_config()
        sdr = RtlSdr()
        sdr.sample_rate = sample_rate

        print(f"Starting IQ data collection for {duration} minutes...")
        gather_iq_data_continuous(sdr, ham_bands, freq_step, runs_per_freq, 'collected_iq_data.csv', duration)

    except KeyboardInterrupt:
        print("Data collection interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if sdr is not None:
            sdr.close()
            print("Closed SDR device.")