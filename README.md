# SpectrumAlert

**SpectrumAlert** is a real-time ham radio spectrum monitoring and anomaly detection system. It continuously scans ham radio frequency bands, detects signal anomalies, identifies modulation types, and publishes the results via MQTT for remote monitoring.

## Features
- **Real-time spectrum monitoring** for ham radio bands (6M, 4M, 2M, 70CM, 23CM)
- **Machine learning-powered anomaly detection** using Isolation Forest
- **Modulation type detection** for AM, FM, USB, LSB, QAM, and more
- **Signal strength calculation** in dB
- **MQTT integration** to publish anomalies, modulation types, signal strength, and receiver coordinates
- **Receiver geolocation** support (latitude and longitude)

## Getting Started

### Prerequisites
To use SpectrumAlert, you'll need the following:
- [RTL-SDR](https://www.rtl-sdr.com/) USB device
- Python 3.x
- Required Python packages:
  - `numpy`
  - `scipy`
  - `sklearn`
  - `paho-mqtt`
  - `rtlsdr`

Install the required packages:
```bash
pip install numpy scipy scikit-learn paho-mqtt pyrtlsdr
```

## Installation

Clone the repository:

```
git clone https://github.com/yourusername/spectrumalert.git
cd spectrumalert
```

Set up your MQTT server and replace the MQTT_BROKER and MQTT_PORT values in the script to point to your server.

Ensure that the coordinates of your receiver are updated in the script:



    RECEIVER_LAT = your_latitude
    RECEIVER_LON = your_longitude

### Running SpectrumAlert

Run the script to begin scanning for anomalies and publishing results:

```
python spectrumalert.py
```

## How It Works

SpectrumAlert scans ham radio bands using an RTL-SDR receiver.
It uses machine learning (Isolation Forest) to detect anomalies in the spectrum.
    When an anomaly is detected, it calculates the signal strength and modulation type.
    Results are published to the MQTT server with:
        - Frequency of the anomaly
        - Modulation type
        - Signal strength in dB
        - Receiver coordinates

MQTT Topics

- hamradio/anomalies: Detected frequency with anomalies.
- hamradio/modulation: Detected modulation type.
- hamradio/signal_strength: Signal strength in dB.
- hamradio/coordinates: Receiver coordinates (latitude, longitude).

Example Use Cases

1. Monitor the ham radio spectrum for unusual signals.
2. Integrate the system into a dashboard for real-time anomaly alerts.
3. Use it for remote signal monitoring in emergency communications.

