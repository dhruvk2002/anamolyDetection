# Real-Time Anomaly Detection in Data Stream

This Streamlit app demonstrates real-time anomaly detection using a sliding window approach and the Isolation Forest algorithm. The app generates a synthetic data stream with seasonality, trend, and noise, and uses machine learning to detect anomalies in the data.

## Features
- **Synthetic Data Stream**: The app generates data with customizable seasonality, trend, and noise.
- **Sliding Window Detection**: Anomalies are detected in real-time using a sliding window mechanism.
- **Isolation Forest**: The `IsolationForest` model is used for unsupervised anomaly detection.
- **Interactive Visualization**: The app provides real-time plotting of data and anomalies, updated dynamically.

## Installation

1. Clone the repository:
    ```bash
    git clone <repository_url>
    cd <repository_folder>
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:
    ```bash
    streamlit run anomaly_detection_streamlit.py
    ```

## Dependencies

- **Streamlit**: For building the interactive web app interface.
- **Matplotlib**: For plotting data streams and detected anomalies.
- **scikit-learn**: Used for the Isolation Forest model.
- **Numpy**: For numerical operations.
- **Python 3.x**: Ensure you have Python installed to run the app.

Install all dependencies using:
```bash
pip install streamlit matplotlib scikit-learn numpy

## How to use
