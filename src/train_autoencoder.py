# src/train_autoencoder.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras.optimizers import Adam
from scipy.signal import butter, filtfilt

# -----------------------------
# Parameters
# -----------------------------
DATA_DIR = "data"
MODEL_DIR = "models"
FS = 360  # default sampling frequency (Hz), adjust if needed
EPOCHS = 50
BATCH_SIZE = 32
SEQ_LEN = FS * 5  # 5-second windows

os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Helper functions
# -----------------------------
def load_all_csv(data_dir, preferred_leads=['MLII', 'V5']):
    """
    Load all CSV files and return concatenated ECG signal.
    If preferred lead is not available, pick the first numeric column.
    """
    signals = []
    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(data_dir, file))
            # Find first available preferred lead
            lead_found = False
            for lead in preferred_leads:
                if lead in df.columns:
                    signals.append(df[lead].values)
                    lead_found = True
                    break
            if not lead_found:
                # fallback: pick first numeric column
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    signals.append(df[numeric_cols[0]].values)
                else:
                    print(f"⚠️ Skipping {file}: no numeric columns found.")
    if len(signals) == 0:
        raise ValueError("No valid ECG signals found in the data folder.")
    return np.concatenate(signals)


def bandpass_filter(sig, fs, low=0.5, high=40.0, order=3):
    """Butterworth bandpass filter"""
    nyq = 0.5 * fs
    lowcut = low / nyq
    highcut = high / nyq
    b, a = butter(order, [lowcut, highcut], btype='band')
    return filtfilt(b, a, sig)

def create_sequences(sig, seq_len):
    """Split signal into overlapping sequences for training"""
    sequences = []
    for i in range(0, len(sig) - seq_len + 1, seq_len):
        sequences.append(sig[i:i+seq_len])
    return np.array(sequences)

# -----------------------------
# Load & preprocess data
# -----------------------------
print("Loading CSV files...")
raw_signal = load_all_csv(DATA_DIR)
print(f"Total samples loaded: {len(raw_signal)}")

print("Filtering signal...")
filtered_signal = bandpass_filter(raw_signal, FS)

# Normalize signal
mean_val = np.mean(filtered_signal)
std_val = np.std(filtered_signal)
filtered_signal = (filtered_signal - mean_val) / std_val

# Save normalization stats
np.save(os.path.join(MODEL_DIR, "mean.npy"), np.array([mean_val]))
np.save(os.path.join(MODEL_DIR, "std.npy"), np.array([std_val]))

# Create sequences
X = create_sequences(filtered_signal, SEQ_LEN)
X = X[..., np.newaxis]  # add channel dimension for Conv1D

# Split train/test
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
print(f"Training sequences: {len(X_train)}, Testing sequences: {len(X_test)}")

# -----------------------------
# Build 1D Conv Autoencoder
# -----------------------------
input_layer = Input(shape=(SEQ_LEN,1))

# Encoder
x = Conv1D(32, 7, activation='relu', padding='same')(input_layer)
x = MaxPooling1D(2, padding='same')(x)
x = Conv1D(16, 7, activation='relu', padding='same')(x)
encoded = MaxPooling1D(2, padding='same')(x)

# Decoder
x = Conv1D(16, 7, activation='relu', padding='same')(encoded)
x = UpSampling1D(2)(x)
x = Conv1D(32, 7, activation='relu', padding='same')(x)
x = UpSampling1D(2)(x)
decoded = Conv1D(1, 7, activation='linear', padding='same')(x)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer=Adam(1e-3), loss='mse')
autoencoder.summary()

# -----------------------------
# Train model
# -----------------------------
print("Training autoencoder...")
history = autoencoder.fit(
    X_train, X_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, X_test)
)

# -----------------------------
# Save model
# -----------------------------
model_path = os.path.join(MODEL_DIR, "autoencoder_model.h5")
autoencoder.save(model_path)
print(f"Model saved to {model_path}")
print("Training complete ✅")
