import numpy as np
import os
from src.preprocess import bandpass_filter
from src.predict import detect_anomalies_with_autoencoder

# =========================
# Update these paths to match your system
# =========================
MODEL_DIR = r"C:\ecg_project\models"
model_path = os.path.join(MODEL_DIR, 'autoencoder_model.h5')
mean_path = os.path.join(MODEL_DIR, 'mean.npy')
std_path = os.path.join(MODEL_DIR, 'std.npy')

# Check if files exist
print("Checking files...")
print("Model exists:", os.path.exists(model_path))
print("Mean exists:", os.path.exists(mean_path))
print("Std exists:", os.path.exists(std_path))

if not (os.path.exists(model_path) and os.path.exists(mean_path) and os.path.exists(std_path)):
    raise FileNotFoundError("One or more model/normalization files are missing!")

# Load normalization stats
mean = np.load(mean_path)
std = np.load(std_path)

# =========================
# Create synthetic ECG signal
# =========================
fs = 360  # sampling frequency
t = np.arange(0, 10, 1/fs)  # 10 seconds
signal = 0.3 * np.sin(2 * np.pi * 1.2 * t)  # normal ECG-like sine wave

# Inject clear anomalies
signal[500] += 10  # spike
signal[1500] -= 8  # negative spike

# Bandpass filter
filtered = bandpass_filter(signal, fs)

# Detect anomalies
positions, errors = detect_anomalies_with_autoencoder(filtered, model_path, mean, std, fs=fs)

# Output results
print("\n=== Anomaly Detection Test ===")
print("Number of anomalies detected:", len(positions))
print("Anomaly positions:", positions)
print("First 10 reconstruction errors:", errors[:10])
