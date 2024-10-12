import numpy as np
import configparser
from rtlsdr import RtlSdr
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import joblib
import paho.mqtt.client as mqtt
from scipy.signal import welch
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
        'modulation': config['MQTT']['topic_modulation'],
        'signal_strength': config['MQTT']['topic_signal_strength'],
        'coordinates': config['MQTT']['topic_coordinates']
    }

    return ham_bands, freq_step, sample_rate, runs_per_freq, receiver_lat, receiver_lon, mqtt_broker, mqtt_port, mqtt_topics

# Function to load the pre-trained anomaly detection model
def load_anomaly_detection_model(model_file='anomaly_detection_model.pkl'):
    if os.path.exists(model_file):
        model = joblib.load(model_file)
        print(f"Anomaly detection model loaded from {model_file}")
    else:
        model = IsolationForest(contamination=0.05, random_state=42)
        print("No pre-trained anomaly model found. A new model will be created.")
    return model

# Function to load the pre-trained RF fingerprinting model (placeholder)
def load_rf_fingerprinting_model(model_file='rf_fingerprinting_model.pkl'):
    if os.path.exists(model_file):
        model = joblib.load(model_file)
        print(f"RF fingerprinting model loaded from {model_file}")
    else:
        model = RandomForestClassifier()
        print("No pre-trained RF fingerprinting model found. A new model will be created.")
    return model

def extract_features(iq_data, target_num_features=None):
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

    features = [
        mean_amplitude, std_amplitude, mean_fft_magnitude, std_fft_magnitude,
        skew_amplitude, kurt_amplitude, skew_phase, kurt_phase, cyclo_autocorr
    ]

    # If target_num_features is provided, adjust the features list accordingly
    if target_num_features is not None:
        if len(features) < target_num_features:
            # Pad with zeros if fewer features than expected
            features += [0] * (target_num_features - len(features))
        elif len(features) > target_num_features:
            # Trim features if there are more than expected
            features = features[:target_num_features]

    return features

# Function to calculate signal strength (placeholder)
def calculate_signal_strength(iq_data):
    amplitude = np.abs(iq_data)
    signal_strength_db = 10 * np.log10(np.mean(amplitude**2))
    return signal_strength_db

# MQTT client setup (placeholder)
def setup_mqtt(broker, port):
    client = mqtt.Client()
    client.connect(broker, port, 60)
    return client

def monitor_spectrum(sdr, model, anomaly_model, mqtt_client, ham_bands, freq_step, sample_rate, runs_per_freq, mqtt_topics, receiver_lat, receiver_lon):
    known_features = []
    similarity_threshold = 0.3  # Threshold to consider a device as similar

    # Get the number of features the anomaly_model expects
    try:
        expected_num_features = anomaly_model.estimators_[0].n_features_in_
    except AttributeError:
        # If the model is not yet fitted, you can set a default number or handle it as needed
        expected_num_features = 9  # Default to the current number of features if unknown

    while True:
        for band_start, band_end in ham_bands:
            current_freq = band_start
            while current_freq <= band_end:
                for _ in range(runs_per_freq):
                    sdr.center_freq = current_freq
                    iq_samples = sdr.read_samples(128 * 1024)
                    features = extract_features(iq_samples, target_num_features=expected_num_features)
                    signal_strength_db = calculate_signal_strength(iq_samples)

                    # Detect anomalies
                    is_anomaly = anomaly_model.predict([features])[0] == -1
                    if is_anomaly:
                        print(f"Anomaly detected at {current_freq / 1e6:.2f} MHz with features: {features}")
                        mqtt_client.publish(mqtt_topics['anomalies'], f"Anomaly at {current_freq / 1e6:.2f} MHz")

                    # Update the model with new data if known features exist
                    if len(known_features) > 1:
                        labels = ["Device" for _ in known_features]
                        model.fit(known_features, labels)

                    known_features.append(features)
                    freq_mhz = current_freq / 1e6
                    mqtt_client.publish(mqtt_topics['signal_strength'], f"{signal_strength_db:.2f}")
                    mqtt_client.publish(mqtt_topics['coordinates'], f"Latitude: {receiver_lat}, Longitude: {receiver_lon}")
                    print(f"Monitoring {freq_mhz:.2f} MHz, Signal Strength: {signal_strength_db:.2f} dB")

                current_freq += freq_step

# Main execution
if __name__ == "__main__":
    try:
        # Load the configuration
        ham_bands, freq_step, sample_rate, runs_per_freq, receiver_lat, receiver_lon, mqtt_broker, mqtt_port, mqtt_topics = read_config()

        # Load the pre-trained RF fingerprinting and anomaly detection models
        anomaly_model = load_anomaly_detection_model('anomaly_detection_model.pkl')
        rf_model = load_rf_fingerprinting_model('rf_fingerprinting_model.pkl')

        # Instantiate RTL-SDR
        sdr = RtlSdr()
        sdr.sample_rate = sample_rate
        sdr.gain = 'auto'

        # Setup MQTT client
        mqtt_client = setup_mqtt(mqtt_broker, mqtt_port)
        mqtt_client.loop_start()

        # Monitor the ham bands for anomalies and report results to MQTT
        monitor_spectrum(sdr, rf_model, anomaly_model, mqtt_client, ham_bands, freq_step, sample_rate, runs_per_freq, mqtt_topics, receiver_lat, receiver_lon)

    except KeyboardInterrupt:
        print("Monitoring stopped by user.")
        sdr.close()
        mqtt_client.disconnect()
        print("Closed SDR device and disconnected from MQTT.")
