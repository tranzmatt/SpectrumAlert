import numpy as np
import configparser
import os
import csv
import time
import sys
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import SoapySDR
from SoapySDR import Device as SoapyDevice

SoapySDR.SoapySDR_setLogLevel(SoapySDR.SOAPY_SDR_WARNING)  # Suppress INFO messages

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
    sdr_type = config['GENERAL'].get('sdr_type', 'rtlsdr')

    return ham_bands, freq_step, sample_rate, runs_per_freq, sdr_type


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
    
    #print(f"Data saved to {filename}")

# Function to gather IQ data and process with reduced features
def gather_data_lite(sdr_type, ham_bands, freq_step, runs_per_freq, filename, duration_minutes):

    sdr = None
    try:

        # Enumerate available SDR devices
        device_dicts = [dict(dev) for dev in SoapyDevice.enumerate()]
        device_list = [dev for dev in device_dicts if dev['driver'] == sdr_type]
        device_count = len(device_list)

        if device_count == 0:
            raise RuntimeError(f"No available SDR devices of type {sdr_type}")
        else:
            print(f"Found {device_count} devices of type {sdr_type}")
            for i, dev in enumerate(device_list):
                print(f"Device {i}: {dev}")

        # ✅ Assign IDs correctly: `serial` for RTL-SDR, `index` for others
        if sdr_type == "rtlsdr":
            device_ids = [dev['serial'] for dev in device_list]  # Use serials for RTL-SDR
        else:
            device_ids = [str(i) for i in range(device_count)]  # Use indexes for other SDRs


        # ✅ Use `serial` for RTL-SDR, `index` for others
        if sdr_type == "rtlsdr":
            sdr = SoapySDR.Device(dict(driver=sdr_type, serial=device_ids[0]))
            gain_value = 10.0  # Adjust gain manually
            sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, gain_value)
        else:
            sdr = SoapySDR.Device(dict(driver=sdr_type, index=str(device_ids[0])))
            sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, 'auto')

        header_written = False
        start_time = time.time()
        duration_seconds = duration_minutes * 60  # Convert minutes to seconds

        # Initialize IsolationForest for anomaly detection
        anomaly_detector = IsolationForest(contamination=0.05, random_state=42)

        # Fit PCA on some initial data
        pca_training_data = []
        for band_start, band_end in ham_bands:

            sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, band_start)
            buff = np.zeros(LITE_SAMPLE_SIZE, dtype=np.complex64)
            stream = sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
            sdr.activateStream(stream)
            sr = sdr.readStream(stream, [buff], len(buff))

            if sr.ret > 0:  # ✅ Only process valid samples
                iq_samples = buff[:sr.ret]  # ✅ Extract only valid IQ samples
                features = extract_features(iq_samples)
                pca_training_data.append(features)

                pca = PCA(n_components=min(2, len(pca_training_data[0]), len(pca_training_data)))  # Only 2 features
                pca.fit(pca_training_data)

        stream = None
        while time.time() - start_time < duration_seconds:
            for band_start, band_end in ham_bands:
                current_freq = band_start
                while current_freq <= band_end:
                    run_features = []
                    for _ in range(runs_per_freq):

                        sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, current_freq)
                        buff = np.zeros(LITE_SAMPLE_SIZE, dtype=np.complex64)
                        stream = sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
                        sdr.activateStream(stream)
                        sr = sdr.readStream(stream, [buff], len(buff))

                        if sr.ret > 0:  # ✅ Only process valid samples
                            iq_samples = buff[:sr.ret]  # ✅ Extract only valid IQ samples
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

        sdr.closeStream(stream)
        sdr.close()
        print("Closed SDR device and disconnected from MQTT.")

    except KeyboardInterrupt:
        sdr.closeStream(stream)
        sdr.close()
        print("Closed SDR device and disconnected from MQTT.")
        sys.exit(0)



# Main execution
if __name__ == "__main__":
    try:
        # Read configuration
        ham_bands, freq_step, sample_rate, runs_per_freq, sdr_type = read_config('Trainer/config.ini')

        # Get the duration for data gathering from user input
        duration = input("Enter the duration for data gathering (in minutes): ")
        duration = float(duration)

        # Start data gathering
        gather_data_lite(sdr_type, ham_bands, freq_step, runs_per_freq, 'collected_data_lite.csv', duration)

    except KeyboardInterrupt:
        sys.exit(0)

    except Exception as e:
        print(f"An error occurred: {e}")