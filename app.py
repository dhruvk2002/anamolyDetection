import random
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from collections import deque
import streamlit as st

# Generate a synthetic data stream with seasonality, trend, and noise
def generate_data_stream(length, seasonality=7, trend=0.01, noise=0.1):
    for i in range(length):
        seasonal_component = math.sin(2 * math.pi * i / seasonality)
        trend_component = i * trend
        noise_component = random.gauss(0, noise)
        yield seasonal_component + trend_component + noise_component

# Sliding window anomaly detection using Isolation Forest
def detect_anomalies_in_window(window_data, contamination=0.01):
    model = IsolationForest(contamination=contamination)
    model.fit(window_data.reshape(-1, 1))
    predictions = model.predict(window_data.reshape(-1, 1))
    return predictions == -1  # Return boolean array of anomalies

# Streamlit real-time visualization with sliding window updates
def real_time_visualization(data_stream, window_size=100, contamination=0.01):
    st.title("Real-Time Anomaly Detection in Data Stream")

    fig, ax = plt.subplots(figsize=(12, 6))
    line, = ax.plot([], [], lw=2, label="Data Stream")
    scatter = ax.scatter([], [], color='red', label="Anomalies")

    ax.set_title('Real-Time Data Stream with Anomalies')
    ax.set_xlim(0, window_size)
    ax.set_ylim(-3, 10)
    ax.legend()

    # Sliding window for real-time anomaly detection
    data_window = deque(maxlen=window_size)
    anomalies_indices = deque(maxlen=window_size)
    anomalies_values = deque(maxlen=window_size)

    for i, point in enumerate(data_stream):
        data_window.append(point)

        if len(data_window) == window_size:
            window_data = np.array(data_window)
            anomaly_flags = detect_anomalies_in_window(window_data, contamination)

            # Record anomalies in the current window
            anomalies_in_window = np.where(anomaly_flags)[0]
            anomalies_indices.extend(i - window_size + anomalies_in_window)
            anomalies_values.extend(window_data[anomalies_in_window])

            # Update the plot
            line.set_data(np.arange(i - window_size + 1, i + 1), window_data)
            scatter.set_offsets(np.c_[anomalies_indices, anomalies_values])

            ax.set_xlim(i - window_size, i + 1)
            ax.set_ylim(min(window_data) - 1, max(window_data) + 1)

            # Display the updated plot
            st.pyplot(fig)

# Main execution for Streamlit
if __name__ == "__main__":
    window_size = st.sidebar.slider("Window Size", min_value=50, max_value=500, value=100)
    contamination = st.sidebar.slider("Contamination", min_value=0.001, max_value=0.1, value=0.01)
    data_length = st.sidebar.slider("Data Length", min_value=500, max_value=5000, value=1000)

    st.sidebar.write("Adjust parameters using the sliders.")
    
    data_stream = generate_data_stream(data_length)
    real_time_visualization(data_stream, window_size=window_size, contamination=contamination)
