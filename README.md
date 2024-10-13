Here’s a **README.md** file for your software that includes installation instructions, use cases, example scenarios, and explanations of the software’s capabilities, including its ability to detect anomalies and track suspected devices for direction finding.

### README.md

---

# Spectrum Alert

**Spectrum Alert** is a software suite designed to monitor ham radio frequencies for anomalies, perform radio frequency (RF) fingerprinting, and provide real-time insights into radio spectrum usage. This tool can be deployed on multiple devices to perform direction finding and detect illegal or unauthorized transmissions.

## Table of Contents
- [Spectrum Alert](#spectrum-alert)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Installation](#installation)
    - [Requirements:](#requirements)
    - [Installation Steps:](#installation-steps)
  - [Usage](#usage)
    - [Menu Options:](#menu-options)
    - [Use Cases:](#use-cases)
    - [Example Scenarios:](#example-scenarios)
      - [Scenario 1: Simple Spectrum Monitoring](#scenario-1-simple-spectrum-monitoring)
      - [Scenario 2: Automated Process](#scenario-2-automated-process)
      - [Scenario 3: Multi-device Direction Finding](#scenario-3-multi-device-direction-finding)
  - [Anomaly Detection and Heatmap](#anomaly-detection-and-heatmap)
    - [Heatmap of Suspected Devices and Anomalies:](#heatmap-of-suspected-devices-and-anomalies)
    - [License](#license)

## Features
- **RF Spectrum Monitoring**: Continuously scans configured ham bands to detect anomalies in RF signal characteristics.
- **Data Gathering**: Collects IQ (In-phase and Quadrature) data from an RTL-SDR device, which can be used for analysis and model training.
- **Anomaly Detection**: Uses machine learning (Isolation Forest) to detect anomalies in radio signal characteristics.
- **RF Fingerprinting**: Trains models to identify and differentiate between various radio devices based on unique signal features.
- **Real-time Monitoring**: Publishes real-time signal strength, anomalies, and other metrics via MQTT for remote monitoring.
- **Lite Versions**: Optimized versions for Raspberry Pi and other low-resource devices.
- **Automated Process**: A single workflow that automates data gathering, model training, and spectrum monitoring.
- **Direction Finding**: When deployed on multiple devices, Spectrum Alert can help triangulate the source of illegal or unauthorized transmissions by analyzing anomalies detected across multiple receivers.

## Installation

### Requirements:
- Python 3.x
- RTL-SDR device
- Required Python packages: `numpy`, `scikit-learn`, `rtlsdr`, `configparser`, `paho-mqtt`, `joblib`
  
### Installation Steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/slayingripper/spectrum-alert.git
   cd spectrum-alert
   ```

2. Install required Python packages:
   ```bash
   pip install numpy scikit-learn rtlsdr configparser paho-mqtt joblib
   ```

3. Connect your RTL-SDR device to your system.

4. Configure the `config.ini` file located in the `Trainer/` folder. Define ham bands, sample rates, and MQTT settings.

## Usage

The software provides a menu-driven interface that allows you to choose between data gathering, model training, real-time spectrum monitoring, or an automated end-to-end workflow.

### Menu Options:

1. **Gather Data**: Collect RF data from the ham bands for further processing or training.
2. **Train Model**: Train RF fingerprinting and anomaly detection models using previously gathered data.
3. **Monitor Spectrum**: Real-time monitoring of spectrum activity, identifying anomalies and reporting them via MQTT.
4. **Automated Workflow**: Gathers data, trains models, and starts the monitor in a single automated flow.
5. **Check for Existing Data or Models**: Automates the process by skipping steps if data or models already exist.
6. **Start from Scratch**: Deletes all existing datasets and models, allowing you to begin fresh.

### Use Cases:
- **RF Spectrum Surveillance**: Continuously monitor ham bands and automatically detect anomalies that could indicate illegal transmissions.
- **Direction Finding**: Deploy multiple devices in different locations to triangulate the source of illegal transmissions by correlating anomalies across devices.
- **RF Fingerprinting**: Use the software to build models that identify specific devices based on their unique RF fingerprints. This can help track and identify persistent offenders in the spectrum.
- **Automated Monitoring and Detection**: Run the software in a fully automated mode to gather data, train models, and start monitoring without manual intervention.

### Example Scenarios:

#### Scenario 1: Simple Spectrum Monitoring
1. **Step 1**: Run the software and select `Gather Data`.
2. **Step 2**: After gathering data, train the model by selecting `Train Model`.
3. **Step 3**: Start real-time spectrum monitoring by selecting `Monitor Spectrum`.
4. **Step 4**: Detected anomalies will be logged and published to an MQTT broker for remote monitoring.

#### Scenario 2: Automated Process
1. **Step 1**: Select the `Automated Workflow` option.
2. **Step 2**: The software automatically gathers data, trains models, and begins real-time spectrum monitoring.
3. **Step 3**: Anomalies are detected in real-time, and data is published to an MQTT broker.

#### Scenario 3: Multi-device Direction Finding
1. **Step 1**: Deploy multiple devices running **Spectrum Alert** in different locations.
2. **Step 2**: Configure each device to monitor the same ham bands.
3. **Step 3**: Each device reports detected anomalies via MQTT.
4. **Step 4**: Correlate the detected anomalies across multiple devices to triangulate the source of the transmission. Use this to identify and track illegal transmissions or unauthorized users.

## Anomaly Detection and Heatmap

One of the core features of **Spectrum Alert** is anomaly detection. The software uses machine learning models to detect anomalies in the signal's characteristics, which could indicate unauthorized or illegal transmissions.

### Heatmap of Suspected Devices and Anomalies:
- **Heatmap**: Detected anomalies and suspected devices can be used to generate a heatmap showing the geographical concentration of anomalies. When multiple devices are deployed, they can collectively contribute data to enhance the accuracy of anomaly detection and location triangulation.
- **Direction Finding**: By analyzing the anomalies detected by multiple geographically distributed receivers, the system can triangulate the position of unauthorized transmissions, helping you locate the source of illegal or suspicious activity in the RF spectrum.

This feature makes **Spectrum Alert** a powerful tool for authorities or ham radio enthusiasts who want to identify and take action against illegal spectrum users.

### License

The software is released under the GNU General Public License v3.0. You are free to use, modify, and distribute the software under the terms of the license.