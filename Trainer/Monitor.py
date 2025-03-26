import argparse
import configparser
import datetime
import json
import os
import re
import subprocess
import sys
import time

import SoapySDR
import gpsd
import joblib
import numpy as np
import paho.mqtt.client as mqtt
from SoapySDR import Device as SoapyDevice
from sklearn.ensemble import RandomForestClassifier, IsolationForest

SoapySDR.SoapySDR_setLogLevel(SoapySDR.SOAPY_SDR_WARNING)  # Suppress INFO messages


def get_gps_coordinates():
    """
    Retrieves GPS coordinates from gpsd if GPS_SOURCE is set to 'gpsd'.
    Returns (latitude, longitude, altitude) or (None, None, None) if unavailable.
    """
    GPS_SOURCE = os.getenv("GPS_SOURCE", "fixed").lower()

    if GPS_SOURCE == "fixed":
        GPS_FIX_ALT = os.getenv("GPS_FIX_ALT", 1)
        GPS_FIX_LAT = os.getenv("GPS_FIX_LAT", 0)
        GPS_FIX_LON = os.getenv("GPS_FIX_LON", 0)
        print(f"Returning fixed GPS of {GPS_FIX_LAT}, {GPS_FIX_LON}, {GPS_FIX_ALT}")
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


def setup_mqtt_client():
    """
    Initializes and configures the MQTT client using environment variables.
    Returns a connected MQTT client instance and the MQTT topic.
    If an error occurs, returns None, None.
    """
    try:
        # ✅ Load environment variables
        MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
        MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
        MQTT_USER = os.getenv("MQTT_USER", None)
        MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", None)
        MQTT_TOPIC = os.getenv("MQTT_TOPIC", "spectrum/anomaly")

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

        return mqtt_client, MQTT_TOPIC  # ✅ Return client

    except Exception as e:
        print(f"❌ MQTT Setup Error: {e}")
        return None, None  # ✅ Ensure `None` is returned on error


# Function to read and parse the config file
def read_config(config_file='Trainer/config.ini'):

    config = configparser.ConfigParser()

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file '{config_file}' not found.")

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
    sdr_type = config['GENERAL'].get('sdr_type', 'rtlsdr')
    min_db = float(config['GENERAL']['min_db'])

    return ham_bands, freq_step, sample_rate, runs_per_freq, sdr_type, min_db, gain_value


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
    amplitude = np.sqrt(I ** 2 + Q ** 2)
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
    cyclo_autocorr = np.abs(np.correlate(amplitude, amplitude, mode='full')[len(amplitude) // 2:]).mean()

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
    signal_strength_db = 10 * np.log10(np.mean(amplitude ** 2))
    return signal_strength_db


def monitor_spectrum(sdr_type, model, anomaly_model, ham_bands, freq_step, sample_rate,
                     runs_per_freq, min_db, mqtt_client, mqtt_topic):

    device_name = get_device_name()

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
        try:
            gain_names = sdr.listGains(SoapySDR.SOAPY_SDR_RX, 0)
            for name in gain_names:
                sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, name, 20.0)
        except Exception as e:
            print(f"Warning: Failed to set gain: {e}")

    known_features = []
    similarity_threshold = 0.3  # Threshold to consider a device as similar

    # Get the number of features the anomaly_model expects
    try:
        expected_num_features = anomaly_model.estimators_[0].n_features_in_
    except AttributeError:
        # If the model is not yet fitted, you can set a default number or handle it as needed
        expected_num_features = 9  # Default to the current number of features if unknown

    stream = sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
    sdr.activateStream(stream)

    while True:
        for band_start, band_end in ham_bands:
            current_freq = band_start
            while current_freq <= band_end:
                for _ in range(runs_per_freq):
                    sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, current_freq)
                    buff = np.zeros(128 * 1024, dtype=np.complex64)
                    sr = sdr.readStream(stream, [buff], len(buff))

                    if sr.ret > 0:  # ✅ Only process valid samples
                        iq_samples = buff[:sr.ret]  # ✅ Extract only valid IQ samples
                        features = extract_features(iq_samples, target_num_features=expected_num_features)
                        signal_strength_db = calculate_signal_strength(iq_samples)

                        if signal_strength_db < min_db:
                            continue

                        # Detect anomalies
                        is_anomaly = anomaly_model.predict([features])[0] == -1
                        if is_anomaly:
                            freq_data = {}
                            print(
                                f"Anomaly detected at {current_freq / 1e6:.2f} MHz at {signal_strength_db:.2f} dB at with features: {features}")
                            freq_data['anomaly_freq_mhz'] = (current_freq / 1e6)
                            freq_data['signal_strength'] = signal_strength_db

                            # ✅ Get UTC detection time
                            detection_time = datetime.datetime.utcnow().isoformat() + "Z"  # Add "Z" for UTC format
                            latitude, longitude, altitude = get_gps_coordinates()

                            mqtt_payload = {
                                "device": device_name,
                                "detection_time": detection_time,
                                "gps": {"lat": latitude, "lon": longitude, "alt": altitude},
                                "data": freq_data
                            }

                            mqtt_payload_str = json.dumps(mqtt_payload)

                            if mqtt_client:
                                try:
                                    publish_info = None
                                    publish_info: mqtt.MQTTMessageInfo = mqtt_client.publish(mqtt_topic,
                                                                                             mqtt_payload_str)
                                    if publish_info.rc is not None:
                                        publish_info.wait_for_publish(timeout=10)
                                        print(f"📤 Published to MQTT topic '{mqtt_topic}':\n{mqtt_payload_str}")
                                    else:
                                        print(f"⚠️ MQTT Publish failed: No response received.")
                                except Exception as e:
                                    print(f"❌ MQTT Publishing Error: {e}")
                                    print(f"🔄 Trying to re-establish MQTT connection...")
                                    try:
                                        mqtt_client.reconnect()
                                        time.sleep(2)  # Allow time for reconnection
                                    except Exception as recon_error:
                                        print(f"⚠️ MQTT Reconnect Failed: {recon_error}")

                        # Update the model with new data if known features exist
                        if len(known_features) > 1:
                            labels = ["Device" for _ in known_features]
                            model.fit(known_features, labels)

                        known_features.append(features)

                    else:
                        print(f"Warning: No samples received at {current_freq} Hz")

                    current_freq += freq_step


# Main execution
if __name__ == "__main__":

    mqtt_client = None

    try:
        # ✅ Use argparse for command-line parsing
        parser = argparse.ArgumentParser(description="Spectrum Monitoring with SDR.")
        parser.add_argument("-c", "--config", type=str, default="Trainer/config.ini",
                            help="Path to the configuration file (default: Trainer/config.ini)")
        parser.add_argument("-t", "--duration", type=float, default=10,
                            help="Duration in minutes (default: 10)")
        parser.add_argument("-a", "--amodel", type=str, default="anomaly_detection_model_lite.pkl",
                            help="Path to the machine learning model (default: anomaly_detection_model_lite.pkl)")
        parser.add_argument("-r", "--rfmodel", type=str, default="rf_fingerprinting_model.pkl",
                            help="Path to the machine learning model (default: rf_fingerprinting_model.pkl)")

        args = parser.parse_args()

        # ✅ Extract arguments
        config_file = args.config
        duration = args.duration
        amodel_file = args.amodel
        rf_model_file = args.rfmodel

        ham_bands, freq_step, sample_rate, runs_per_freq, sdr_type, min_db, gain_value = read_config(config_file)

        # Load the pre-trained RF fingerprinting and anomaly detection models
        anomaly_model = load_anomaly_detection_model(amodel_file)
        rf_model = load_rf_fingerprinting_model(rf_model_file)

        # Setup MQTT client
        mqtt_client, mqtt_topic = setup_mqtt_client()

        if mqtt_client is None:
            print("❌ MQTT client initialization failed. Exiting...")
            sys.exit(1)  # Exit the script with a non-zero status to indicate failure

        # Monitor the ham bands for anomalies and report results to MQTT
        monitor_spectrum(sdr_type, rf_model, anomaly_model, ham_bands, freq_step, sample_rate, runs_per_freq,
                         min_db, gain_value, mqtt_client, mqtt_topic)

    except KeyboardInterrupt:
        print("Monitoring stopped by user.")
        if mqtt_client:
            mqtt_client.disconnect()
        print("Closed SDR device and disconnected from MQTT.")
