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
- **Python 3.12.5**: Ensure you have Python installed to run the app.

Install all dependencies using:
```bash
pip install streamlit matplotlib scikit-learn numpy
```
## How to use
1. **Adjust Parameters**: Use the slide bar to adjust the parameters
- **Window Size**: The size of the sliding window used for anomaly detection.
- **Contamination**: The proportion of anomalies in the data.
- **Data Length**: The total length of the generated data stream.

2. **View realtime detection**: The app will display the data stream with detected anomalies highlighted in red. The plot updates dynamically as the data stream progresses.

## Future Improvements
- Adding more advanced detection models.
- Extending the visualization to include additional data characteristics such as drift or spikes.
- Improving performance for large data streams.
