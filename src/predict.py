# src/predict.py
import os
import numpy as np
from tensorflow.keras.models import load_model

def sliding_windows_centers(signal_len, window_size, step):
    centers = []
    for start in range(0, max(1, signal_len - window_size + 1), step):
        centers.append(start + window_size // 2)
    return centers

def merge_consecutive_indices(idxs, min_group=2):
    """
    Merge list of integer indices into groups of consecutive runs;
    return the center index of each run that has length >= min_group.
    """
    if len(idxs) == 0:
        return []
    idxs = np.sort(np.unique(idxs))
    groups = []
    cur = [idxs[0]]
    for i in idxs[1:]:
        if i == cur[-1] + 1:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    centers = []
    for g in groups:
        if len(g) >= min_group:
            centers.append(int(np.round(np.mean(g))))
    return centers

def detect_anomalies_with_autoencoder(signal, model_path, mean, std,
                                      fs=360, std_multiplier=3.0,
                                      min_consecutive_windows=2,
                                      inject_test_spike=False):
    """
    Robust anomaly detection using an autoencoder:
      - infer window size from model input shape
      - build overlapping windows (step = window//2)
      - normalize using provided mean/std (scalars or arrays)
      - compute reconstruction errors
      - smooth errors with a small moving average
      - compute robust threshold using median + k * MAD
      - require at least `min_consecutive_windows` consecutive windows above threshold to mark an event

    Returns dict with:
      window_size, step, errors (per-window), threshold, anomaly_positions (sample indices), anomaly_errors
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = load_model(model_path, compile=False)

    # infer window size from model input (None, window_size, 1)
    input_shape = model.input_shape
    if input_shape is None or len(input_shape) < 3:
        raise ValueError("Model has unexpected input shape: " + str(input_shape))
    window_size = int(input_shape[1])
    step = window_size // 2

    sig = np.asarray(signal).copy()
    if inject_test_spike and len(sig) > 600:
        idx = min(500, len(sig)-20)
        sig[idx:idx+10] += np.max(np.abs(sig)) * 2.0

    # pad short signals
    if len(sig) < window_size:
        pad_len = window_size - len(sig)
        sig = np.pad(sig, (0, pad_len), 'constant')

    # build windows
    windows = []
    centers = []
    for start in range(0, len(sig) - window_size + 1, step):
        windows.append(sig[start:start+window_size])
        centers.append(start + window_size // 2)
    windows = np.array(windows)  # shape (n_windows, window_size)
    if windows.size == 0:
        return {
            "window_size": window_size,
            "step": step,
            "errors": np.array([]),
            "threshold": None,
            "anomaly_positions": [],
            "anomaly_errors": [],
            "n_windows": 0
        }

    # reshape for model
    windows = windows[..., np.newaxis]  # (n_windows, window_size, 1)

    # normalize (mean/std may be scalars or arrays broadcastable)
    mean_arr = np.asarray(mean)
    std_arr = np.asarray(std)
    # broadcasting works with numpy arrays; add small eps
    windows_norm = (windows - mean_arr) / (std_arr + 1e-9)

    # predict reconstruction
    recon = model.predict(windows_norm, verbose=0)

    # per-window MSE
    errors = np.mean((recon - windows_norm) ** 2, axis=(1, 2))  # shape (n_windows,)

    # smooth errors with a short moving average to reduce spurious peaks
    k = max(1, int(round(3 * (window_size / fs))))  # ~3 seconds worth or min 1
    if k % 2 == 0:
        k += 1
    if len(errors) >= k:
        # simple moving average
        kernel = np.ones(k) / k
        errors_smooth = np.convolve(errors, kernel, mode='same')
    else:
        errors_smooth = errors.copy()

    # robust threshold: median + std_multiplier * MAD
    median_err = np.median(errors_smooth)
    mad = np.median(np.abs(errors_smooth - median_err)) + 1e-12
    # convert MAD to approximate STD: std ~ 1.4826*MAD (but we simply use MAD)
    threshold = float(median_err + std_multiplier * mad)

    # mark windows above threshold
    above_idx = np.where(errors_smooth > threshold)[0]

    # require consecutive windows to reduce false positives
    event_window_indices = merge_consecutive_indices(above_idx, min_group=min_consecutive_windows)

    # map window indices back to sample centers
    anomaly_centers_samples = []
    anomaly_errs = []
    for widx in event_window_indices:
        if widx < len(centers):
            anomaly_centers_samples.append(int(centers[widx]))
            anomaly_errs.append(float(errors[widx]))

    return {
        "window_size": window_size,
        "step": step,
        "errors": errors_smooth.tolist(),
        "threshold": threshold,
        "anomaly_positions": anomaly_centers_samples,
        "anomaly_errors": anomaly_errs,
        "n_windows": len(errors_smooth)
    }
