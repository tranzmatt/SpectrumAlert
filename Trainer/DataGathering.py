import configparser
import os
import csv
import time
import sys
import threading
import numpy as np
import SoapySDR
from sklearn.decomposition import PCA
from concurrent.futures import ThreadPoolExecutor

from SoapySDR import Device as SoapyDevice
SoapySDR.SoapySDR_setLogLevel(SoapySDR.SOAPY_SDR_WARNING)  # Suppress INFO messages

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
    sdr_type = config['GENERAL'].get('sdr_type', 'rtlsdr')

    return ham_bands, freq_step, sample_rate, runs_per_freq, sdr_type

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



# Function to scan a single band with a specific SDR device
def scan_band(device_id, sdr_type, band_start, band_end, freq_step, runs_per_freq, filename, pca):
    """Scans a frequency band using a specific SDR device."""

    print(f"[Thread-{device_id}] Opening SDR Device {device_id}")

    # ✅ Use `serial` for RTL-SDR, `index` for others
    if sdr_type == "rtlsdr":
        sdr = SoapySDR.Device(dict(driver=sdr_type, serial=device_id))
    else:
        sdr = SoapySDR.Device(dict(driver=sdr_type, index=str(device_id)))

    print(f"[Thread-{device_id}] Opened SDR Device {device_id}")

    sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, 2.048e6)  # Set sample rate
    print(f"[Thread-{device_id}] Setting Sample Rate set")

    try:
        if sdr_type == "rtlsdr":
            # RTL-SDR does not support 'auto' gain, set manually
            gain_value = 10.0  # Adjust gain manually
            print(f"[Thread-{device_id}] Setting gain to {gain_value}")
            sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, gain_value)
        else:
            # Other SDRs like HackRF, LimeSDR, etc.
            sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, 'auto')
            print(f"[Thread-{device_id}] Setting gain to auto")
    except RuntimeError as e:
        print(f"[{sdr_type}] Warning: Gain setting failed: {e}")

    current_freq = band_start

    while current_freq <= band_end:
        run_features = []
        for _ in range(runs_per_freq):
            sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, current_freq)
            print(f"[Thread-{device_id}] Setting frequency {current_freq}")

            # ✅ Read IQ samples using SoapySDR stream API
            buff = np.zeros(256 * 1024, dtype=np.complex64)
            stream = sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
            sdr.activateStream(stream)
            sr = sdr.readStream(stream, [buff], len(buff))
            print(f"[Thread-{device_id}] Stream Read")

            if sr.ret > 0:
                iq_samples = buff[:sr.ret]
                features = extract_features(iq_samples)
                run_features.append(features)
            else:
                print(f"[Thread-{device_id}] Warning: No samples at {current_freq} Hz")

            sdr.deactivateStream(stream)
            sdr.closeStream(stream)
            print(f"[Thread-{device_id}] Stream Closed")

        if run_features:
            avg_features = np.mean(run_features, axis=0)
            with threading.Lock():
                reduced_features = pca.transform([avg_features])  # Thread-safe PCA transformation

            # Save to CSV
            data = [current_freq] + avg_features.tolist()
            save_data_to_csv(data, filename)

        current_freq += freq_step  # Move to next frequency

    print(f"[Thread-{device_id}] Closing SDR Device {device_id}")
    sdr = None  # Release SDR


def gather_iq_data_parallel(sdr_type, ham_bands, freq_step, runs_per_freq, filename, duration_minutes):
    """Manages SDR scanning in parallel using multiple devices."""

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

    # Collect initial data to fit PCA (Uses First Available SDR)
    print(f"gather_iq_data_parallel: Collecting PCA Training Data")
    pca_training_data = []

    sdr = SoapySDR.Device(dict(driver=sdr_type, index="0"))  # Open first device for PCA collection
    sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, 2.048e6)

    try:
        if sdr_type == "rtlsdr":
            # RTL-SDR does not support 'auto' gain, set manually
            gain_value = 10.0  # Adjust gain manually
            sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, gain_value)
            print(f"PCA Collection: Setting gain to {gain_value}")
        else:
            # Other SDRs like HackRF, LimeSDR, etc.
            sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, 'auto')
            print(f"PCA Collection: Setting gain to auto")
    except RuntimeError as e:
        print(f"[{sdr_type}] Warning: Gain setting failed: {e}")

    for band_start, band_end in ham_bands:
        sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, band_start)
        buff = np.zeros(256 * 1024, dtype=np.complex64)
        stream = sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
        sdr.activateStream(stream)
        sr = sdr.readStream(stream, [buff], len(buff))

        if sr.ret > 0:
            iq_samples = buff[:sr.ret]
            features = extract_features(iq_samples)
            pca_training_data.append(features)
        else:
            print(f"Warning: No samples received for {band_start} Hz")

        sdr.deactivateStream(stream)
        sdr.closeStream(stream)
        print(f"PCA Training Data SDR closed")

    sdr = None  # Release PCA SDR

    # Train PCA Model
    num_features = len(pca_training_data[0])
    n_components = min(8, len(pca_training_data), num_features)
    pca = PCA(n_components=n_components)
    pca.fit(pca_training_data)

    print(f"gather_iq_data_parallel: Starting Parallel SDR Scanning")

    # Parallel scanning of bands
    num_threads = min(device_count, len(ham_bands))  # Limit threads to available SDRs

    print(f"Number of threads: {num_threads} (min {device_count} devices vs {len(ham_bands)} ham_bands)")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i, (band_start, band_end) in enumerate(ham_bands):
            device_id = device_ids[i % device_count]  # Assign a unique device (serial for RTL-SDR, index for others)
            print(f"Activating thread {device_id}")
            futures.append(
                executor.submit(scan_band, device_id, sdr_type, band_start, band_end, freq_step, runs_per_freq,
                                filename, pca))

        # Wait for all threads to finish
        for future in futures:
            future.result()


# Main execution
if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            duration = float(sys.argv[1])
        else:
            raise ValueError("No duration specified. Please provide the duration in minutes as an argument.")

        ham_bands, freq_step, sample_rate, runs_per_freq, sdr_type = read_config()

        print(f"Starting IQ data collection for {duration} minutes...")
        gather_iq_data_parallel(sdr_type, ham_bands, freq_step, runs_per_freq, 'collected_iq_data.csv', duration)

    except KeyboardInterrupt:
        print("Data collection interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
