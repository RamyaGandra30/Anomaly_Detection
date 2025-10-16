import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


def load_ecg_csv(path, lead_col='MLII'):
    """
    Load ECG signal and time from a CSV file.

    Parameters:
        path (str): Path to the CSV file.
        lead_col (str): Column name for the ECG lead (default is 'MLII').

    Returns:
        tuple: signal (np.ndarray), time (np.ndarray or None), dataframe (pd.DataFrame)
    """
    df = pd.read_csv(path)
    # Expect columns like: sample_index, time, MLII, etc.
    sig = df[lead_col].values.astype(float)
    time = df['time'].values.astype(float) if 'time' in df.columns else None
    return sig, time, df


def bandpass_filter(signal, fs, lowcut=0.5, highcut=40.0, order=4):
    """
    Apply a Butterworth bandpass filter to the ECG signal.

    Parameters:
        signal (np.ndarray): Input ECG signal.
        fs (float): Sampling frequency in Hz.
        lowcut (float): Low cutoff frequency (Hz).
        highcut (float): High cutoff frequency (Hz).
        order (int): Filter order.

    Returns:
        np.ndarray: Filtered ECG signal.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)
    return filtered


def estimate_fs_from_time(time_arr):
    """
    Estimate the sampling frequency from time array.

    Parameters:
        time_arr (np.ndarray): Array of time values.

    Returns:
        float or None: Estimated sampling frequency in Hz.
    """
    if time_arr is None:
        return None

    # Median sampling interval
    dt = np.median(np.diff(time_arr))
    if dt <= 0:
        return None

    fs = 1.0 / dt
    return float(round(fs))
