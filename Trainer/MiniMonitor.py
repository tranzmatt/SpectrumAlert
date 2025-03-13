import numpy as np
import configparser
from rtlsdr import RtlSdr
from sklearn.ensemble import IsolationForest
import joblib
import paho.mqtt.client as mqtt
import os
import subprocess
import socket
import gpsd
import json
import time

def get_gps_coordinates(receiver_lat, receiver_lon):
    """
    Retrieves GPS coordinates from gpsd if GPS_SOURCE is set to 'gpsd'.
    Returns (latitude, longitude, altitude) or (None, None, None) if unavailable.
    """
    GPS_SOURCE = os.getenv("GPS_SOURCE", "none").lower()

    if GPS_SOURCE == "fixed":
        GPS_FIX_ALT = os.getenv("GPS_FIX_ALT", 1)
        GPS_FIX_LAT = os.getenv("GPS_FIX_LAT", receiver_lat)
        GPS_FIX_LON = os.getenv("GPS_FIX_LON", receiver_lon)
        return GPS_FIX_LAT, GPS_FIX_LON, GPS_FIX_ALT

    if GPS_SOURCE == "gpsd":
        try:
            # ✅ Connect to gpsd
            gpsd.connect(host="localhost", port=2947)

            # ✅ Get GPS data
            gps_data = gpsd.get_current()

            if gps_data is None:
                print("⚠️ No GPS data available. GPS may not be active.")
                return None, None, None

            if gps_data.mode >= 2:  # 2D or 3D fix
                latitude = gps_data.lat
                longitude = gps_data.lon
                altitude = gps_data.alt if gps_data.mode == 3 else None  # Altitude available in 3D mode
                print(f"📍 GPSD Coordinates: {latitude}, {longitude}, Alt: {altitude}m")
                return latitude, longitude, altitude
            else:
                print("⚠️ No GPS fix yet.")
        except Exception as e:
            print(f"❌ GPSD Error: {e}")
    else:
        print("No available gps source")

    return None, None, None  # Return None if GPS is unavailable

def get_primary_mac():
    """Retrieves the primary MAC address (uppercase, no colons)."""
    try:
        # ✅ Get MAC address using `ip link` (Linux)
        mac_output = subprocess.check_output("ip link show | grep -m 1 'link/ether' | awk '{print $2}'",
                                             shell=True, text=True).strip()

        # ✅ Remove colons and convert to uppercase
        mac_clean = re.sub(r'[:]', '', mac_output).upper()

        return mac_clean
    except Exception as e:
        print(f"❌ Error getting MAC address: {e}")
        return "UNKNOWNMAC"


def get_device_name():
    """
    Retrieves the device name from environment variable.
    If unavailable, falls back to 'uname -m' + 'hostname'.
    """

    # If on Balena
    device_name = os.getenv("BALENA_DEVICE_NAME_AT_INIT")

    if not device_name:
        try:
            host = subprocess.check_output("hostname", shell=True, text=True).strip()
            mac = get_primary_mac()  # ✅ Get the primary MAC address
            device_name = f"{host}-{mac}"  # ✅ Append MAC address
        except Exception as e:
            print(f"❌ Error getting fallback device name: {e}")
            device_name = "unknown-device"

    return device_name


def setup_mqtt_client(mqtt_broker, mqtt_port):
    """
    Initializes and configures the MQTT client using environment variables.
    Returns a connected MQTT client instance and the MQTT topic.
    If an error occurs, returns None, None.
    """
    try:
        # ✅ Load environment variables
        MQTT_BROKER = os.getenv("MQTT_BROKER", mqtt_broker)
        MQTT_PORT = int(os.getenv("MQTT_PORT", mqtt_port))
        MQTT_USER = os.getenv("MQTT_USER", None)
        MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", None)

        # ✅ TLS & CA Certificate Options
        MQTT_TLS = int(os.getenv("MQTT_TLS", 0))  # 1 = Enable TLS, 0 = Disable
        MQTT_USE_CA_CERT = int(os.getenv("MQTT_USE_CA_CERT", 0))  # 1 = Use CA Cert, 0 = Disable
        MQTT_CA_CERT = os.getenv("MQTT_CA_CERT", "/path/to/ca.crt")  # Path to CA Cert

        print(f"📡 Configuring MQTT: {MQTT_BROKER}:{MQTT_PORT} (TLS: {MQTT_TLS}, CA Cert: {MQTT_USE_CA_CERT})")

        # ✅ Create MQTT client
        mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

        # ✅ Enable automatic reconnect
        mqtt_client.reconnect_delay_set(min_delay=1, max_delay=30)

        # ✅ Use TLS if enabled
        if MQTT_TLS:
            print("🔐 Enabling TLS for MQTT...")
            mqtt_client.tls_set(ca_certs=MQTT_CA_CERT if MQTT_USE_CA_CERT else None)

        # ✅ Define callback functions for connection management
        def on_connect(client, userdata, flags, rc, properties):
            if rc == 0:
                print("✅ MQTT Connected Successfully!")
            else:
                print(f"⚠️ MQTT Connection Failed with Code {rc}")

        def on_disconnect(client, userdata, rc, *args):
            print("❌ MQTT on_disconnect! Trying to reconnect...")
            try:
                client.reconnect()
            except Exception as e:
                print(f"⚠️ MQTT Reconnect Failed: {e}")

        mqtt_client.on_connect = on_connect
        mqtt_client.on_disconnect = on_disconnect

        # ✅ Set username/password if provided
        mqtt_client.username_pw_set(MQTT_USER, MQTT_PASSWORD)

        # ✅ Connect to MQTT broker
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        print("✅ Connected to MQTT broker successfully!")

        return mqtt_client  # ✅ Return client

    except Exception as e:
        print(f"❌ MQTT Setup Error: {e}")
        return None, None  # ✅ Ensure `None` is returned on error


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

    return (ham_bands, freq_step, sample_rate, runs_per_freq, receiver_lat, receiver_lon,
            mqtt_broker, mqtt_port, mqtt_topics)

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

def monitor_spectrum_lite(sdr, anomaly_model, ham_bands, freq_step, sample_rate,
                          runs_per_freq, mqtt_client, mqtt_topics, receiver_lat, receiver_lon):
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
        (ham_bands, freq_step, sample_rate, runs_per_freq, receiver_lat, receiver_lon,
         mqtt_broker, mqtt_port, mqtt_topics) = read_config()

        # Load the pre-trained anomaly detection model
        anomaly_model = load_anomaly_detection_model('anomaly_detection_model_lite.pkl')

        # Instantiate RTL-SDR
        sdr = RtlSdr()
        sdr.sample_rate = sample_rate
        sdr.gain = 'auto'

        # Setup MQTT client
        mqtt_client = setup_mqtt_client(mqtt_broker, mqtt_port)
        if mqtt_client is None:
            print("⚠️ MQTT setup failed. Skipping MQTT publishing.")

        # Monitor the ham bands for anomalies and report results to MQTT
        monitor_spectrum_lite(sdr, anomaly_model, ham_bands, freq_step, sample_rate, runs_per_freq,
                              mqtt_client, mqtt_topics, receiver_lat, receiver_lon)

    except KeyboardInterrupt:
        print("Monitoring stopped by user.")
        sdr.close()
        mqtt_client.disconnect()
        print("Closed SDR device and disconnected from MQTT.")
