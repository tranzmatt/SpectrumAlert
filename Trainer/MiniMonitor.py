import numpy as np
import configparser
from rtlsdr import RtlSdr
from sklearn.ensemble import IsolationForest
import joblib
import paho.mqtt.client as mqtt
import os

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
        'signal_strength': config['MQTT']['topic_signal_strength'],
        'coordinates': config['MQTT']['topic_coordinates']
    }

    return ham_bands, freq_step, sample_rate, runs_per_freq, receiver_lat, receiver_lon, mqtt_broker, mqtt_port, mqtt_topics

# Function to load the pre-trained anomaly detection model
def load_anomaly_detection_model(model_file='anomaly_detection_model_lite.pkl'):
    if os.path.exists(model_file):
        model = joblib.load(model_file)
        print(f"Anomaly detection model loaded from {model_file}")
    else:
        model = IsolationForest(contamination=0.05, random_state=42)
        print("No pre-trained anomaly model found. A new model will be created.")
    return model

# Lite version of feature extraction with 2 features (to match the trained model)
def extract_lite_features(iq_data):
    I = np.real(iq_data)
    Q = np.imag(iq_data)
    amplitude = np.sqrt(I**2 + Q**2)

    # Basic amplitude statistics
    mean_amplitude = np.mean(amplitude)
    std_amplitude = np.std(amplitude)

    # Return only 2 features (matching the trained model)
    return [mean_amplitude, std_amplitude]

# Function to calculate signal strength (simplified)
def calculate_signal_strength(iq_data):
    amplitude = np.abs(iq_data)
    signal_strength_db = 10 * np.log10(np.mean(amplitude**2))
    return signal_strength_db

# MQTT client setup
def setup_mqtt(broker, port):
    client = mqtt.Client()
    client.connect(broker, port, 60)
    return client

def monitor_spectrum_lite(sdr, anomaly_model, mqtt_client, ham_bands, freq_step, sample_rate, runs_per_freq, mqtt_topics, receiver_lat, receiver_lon):
    # Get the number of features the anomaly_model expects
    try:
        expected_num_features = anomaly_model.estimators_[0].n_features_in_
    except AttributeError:
        expected_num_features = 2  # We expect 2 features in the lite version

    while True:
        for band_start, band_end in ham_bands:
            current_freq = band_start
            while current_freq <= band_end:
                for _ in range(runs_per_freq):
                    sdr.center_freq = current_freq
                    iq_samples = sdr.read_samples(64 * 1024)  # Reduced sample size for lite version
                    features = extract_lite_features(iq_samples)
                    signal_strength_db = calculate_signal_strength(iq_samples)

                    # Ensure we only send the correct number of features
                    if len(features) == expected_num_features:
                        is_anomaly = anomaly_model.predict([features])[0] == -1
                        if is_anomaly:
                            print(f"Anomaly detected at {current_freq / 1e6:.2f} MHz")
                            mqtt_client.publish(mqtt_topics['anomalies'], f"Anomaly at {current_freq / 1e6:.2f} MHz")

                        freq_mhz = current_freq / 1e6
                        mqtt_client.publish(mqtt_topics['signal_strength'], f"{signal_strength_db:.2f} dB")
                        mqtt_client.publish(mqtt_topics['coordinates'], f"Latitude: {receiver_lat}, Longitude: {receiver_lon}")
                        print(f"Monitoring {freq_mhz:.2f} MHz, Signal Strength: {signal_strength_db:.2f} dB")

                current_freq += freq_step

# Main execution
if __name__ == "__main__":
    try:
        # Load the configuration
        ham_bands, freq_step, sample_rate, runs_per_freq, receiver_lat, receiver_lon, mqtt_broker, mqtt_port, mqtt_topics = read_config()

        # Load the pre-trained anomaly detection model
        anomaly_model = load_anomaly_detection_model('anomaly_detection_model_lite.pkl')

        # Instantiate RTL-SDR
        sdr = RtlSdr()
        sdr.sample_rate = sample_rate
        sdr.gain = 'auto'

        # Setup MQTT client
        mqtt_client = setup_mqtt(mqtt_broker, mqtt_port)
        mqtt_client.loop_start()

        # Monitor the ham bands for anomalies and report results to MQTT
        monitor_spectrum_lite(sdr, anomaly_model, mqtt_client, ham_bands, freq_step, sample_rate, runs_per_freq, mqtt_topics, receiver_lat, receiver_lon)

    except KeyboardInterrupt:
        print("Monitoring stopped by user.")
        sdr.close()
        mqtt_client.disconnect()
        print("Closed SDR device and disconnected from MQTT.")
