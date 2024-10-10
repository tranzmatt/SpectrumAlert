import numpy as np
import configparser
from scipy.signal import welch
from sklearn.ensemble import IsolationForest
from rtlsdr import RtlSdr
import paho.mqtt.client as mqtt

# Function to read and parse the config file
def read_config(config_file='Trainer/config.ini'):
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

    # Parse receiver settings
    receiver_lat = float(config['RECEIVER']['latitude'])
    receiver_lon = float(config['RECEIVER']['longitude'])

    # Parse MQTT settings
    mqtt_broker = config['MQTT']['broker']
    mqtt_port = int(config['MQTT']['port'])
    mqtt_topics = {
        'anomalies': config['MQTT']['topic_anomalies'],
        'modulation': config['MQTT']['topic_modulation'],
        'signal_strength': config['MQTT']['topic_signal_strength'],
        'coordinates': config['MQTT']['topic_coordinates']
    }

    return ham_bands, freq_step, sample_rate, runs_per_freq, receiver_lat, receiver_lon, mqtt_broker, mqtt_port, mqtt_topics

# Anomaly detection model
anomaly_detector = IsolationForest(contamination=0.01)

# Function to calculate signal strength in dB
def calculate_signal_strength(iq_data):
    power = np.mean(np.abs(iq_data) ** 2)
    return 10 * np.log10(power)

# Function to extract features from IQ data
def extract_features(iq_data):
    I = np.real(iq_data)
    Q = np.imag(iq_data)
    amplitude = np.sqrt(I**2 + Q**2)
    fft_values = np.fft.fft(iq_data)
    fft_magnitude = np.abs(fft_values)
    return [np.mean(amplitude), np.std(amplitude), np.mean(fft_magnitude), np.std(fft_magnitude)]

# Detect anomaly using Isolation Forest
def detect_anomaly(iq_data):
    features = extract_features(iq_data)
    prediction = anomaly_detector.predict([features])
    return prediction == -1

# Function to detect modulation type
def detect_modulation_type(iq_data, sample_rate):
    amplitude = np.abs(iq_data)
    phase = np.unwrap(np.angle(iq_data))
    freq = np.diff(phase) / (2.0 * np.pi) * sample_rate
    freqs, psd = welch(iq_data, fs=sample_rate, nperseg=1024, return_onesided=False)
    if np.std(amplitude) > 0.1 and np.std(freq) < 0.05:
        return "AM"
    if np.std(freq) > 0.05 and np.std(amplitude) < 0.05:
        return "FM"
    if np.std(np.diff(phase)) > 0.1:
        return "PM"
    power_above_carrier = np.sum(psd[freqs > 0])
    power_below_carrier = np.sum(psd[freqs < 0])
    if power_above_carrier > 2 * power_below_carrier:
        return "USB"
    elif power_below_carrier > 2 * power_above_carrier:
        return "LSB"
    return "Unknown"

# Function to scan ham bands and train Isolation Forest model
def scan_spectrum(sdr, ham_bands, freq_step, runs_per_freq):
    normal_data = []
    for band_start, band_end in ham_bands:
        current_freq = band_start
        while current_freq <= band_end:
            run_features = []
            for _ in range(runs_per_freq):
                sdr.center_freq = current_freq
                iq_samples = sdr.read_samples(128 * 1024)
                features = extract_features(iq_samples)
                run_features.append(features)
            avg_features = np.mean(run_features, axis=0)
            normal_data.append(avg_features)
            current_freq += freq_step
    anomaly_detector.fit(normal_data)

# Function to monitor spectrum and publish to MQTT
def monitor_spectrum(sdr, mqtt_client, ham_bands, freq_step, sample_rate, runs_per_freq, mqtt_topics, receiver_lat, receiver_lon):
    while True:
        for band_start, band_end in ham_bands:
            current_freq = band_start
            while current_freq <= band_end:
                run_anomalies = []
                for _ in range(runs_per_freq):
                    sdr.center_freq = current_freq
                    iq_samples = sdr.read_samples(128 * 1024)
                    is_anomaly = detect_anomaly(iq_samples)
                    run_anomalies.append(is_anomaly)
                modulation_type = detect_modulation_type(iq_samples, sample_rate)
                signal_strength_db = calculate_signal_strength(iq_samples)
                if np.mean(run_anomalies) > 0:
                    freq_mhz = current_freq / 1e6
                    mqtt_client.publish(mqtt_topics['anomalies'], f"{freq_mhz:.2f} MHz")
                    mqtt_client.publish(mqtt_topics['modulation'], modulation_type)
                    mqtt_client.publish(mqtt_topics['signal_strength'], f"{signal_strength_db:.2f} dB")
                    mqtt_client.publish(mqtt_topics['coordinates'], f"Latitude: {receiver_lat}, Longitude: {receiver_lon}")
                current_freq += freq_step

# MQTT setup function
def setup_mqtt(mqtt_broker, mqtt_port):
    client = mqtt.Client()
    client.connect(mqtt_broker, mqtt_port, 60)
    return client

# Main execution
if __name__ == "__main__":
    try:
        ham_bands, freq_step, sample_rate, runs_per_freq, receiver_lat, receiver_lon, mqtt_broker, mqtt_port, mqtt_topics = read_config()
        sdr = RtlSdr()
        sdr.sample_rate = sample_rate
        sdr.gain = 'auto'
        mqtt_client = setup_mqtt(mqtt_broker, mqtt_port)
        mqtt_client.loop_start()
        scan_spectrum(sdr, ham_bands, freq_step, runs_per_freq)
        monitor_spectrum(sdr, mqtt_client, ham_bands, freq_step, sample_rate, runs_per_freq, mqtt_topics, receiver_lat, receiver_lon)
    except KeyboardInterrupt:
        print("Interrupted by user, stopping...")
    finally:
        sdr.close()
        mqtt_client.disconnect()
