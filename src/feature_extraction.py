# src/feature_extraction.py
import numpy as np
from scipy.signal import find_peaks

def detect_r_peaks(signal, fs, distance_sec=0.25, height=None, prominence=None):
    """
    Detect R-peaks in an ECG signal using scipy.find_peaks.

    Parameters
    ----------
    signal : np.ndarray
        1-D ECG signal (should be filtered).
    fs : float
        Sampling frequency in Hz.
    distance_sec : float
        Minimum distance between peaks in seconds (default 0.25s -> 240 bpm max).
    height : float or None
        Minimum height of peaks (in signal units). If None, an automatic threshold is used.
    prominence : float or None
        Minimum prominence of peaks. If None, automatic heuristic used.

    Returns
    -------
    peaks : np.ndarray
        Indices (samples) of detected peaks.
    props : dict
        Properties returned by scipy.find_peaks (e.g., 'peak_heights', 'prominences').
    """
    # Defensive: ensure numpy array
    sig = np.asarray(signal)
    if fs is None or fs <= 0:
        raise ValueError("Sampling frequency 'fs' must be provided and > 0.")

    distance_samples = int(max(1, round(distance_sec * fs)))

    # Automatic height heuristic if not provided: use percentile of signal
    if height is None:
        # Use 60th percentile of absolute signal as a rough min height (tunable)
        height = np.percentile(np.abs(sig), 60)

    # Automatic prominence heuristic if not provided
    if prominence is None:
        prominence = 0.5 * height if height > 0 else None

    # find_peaks expects positive peaks, if signal is inverted we'll take abs for detection but return indices on original
    # However ECG R-peaks are usually positive after preprocessing — use raw signal but allow negative support:
    try:
        peaks, props = find_peaks(sig, distance=distance_samples, height=height, prominence=prominence)
    except Exception:
        # fallback to more permissive detection
        peaks, props = find_peaks(sig, distance=distance_samples)

    return peaks, props


def compute_rr_intervals(peaks, fs):
    """
    Compute RR intervals (seconds) from R-peak indices.

    Parameters
    ----------
    peaks : array-like
        Sample indices of R-peaks.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    rr_seconds : np.ndarray
        RR intervals in seconds (length = len(peaks)-1).
    """
    peaks = np.asarray(peaks)
    if len(peaks) < 2:
        return np.array([], dtype=float)
    rr_samples = np.diff(peaks)
    rr_seconds = rr_samples.astype(float) / float(fs)
    return rr_seconds

def clean_rr_intervals(rr_seconds, min_rr=0.3, max_rr=2.0):
    """
    Remove physiologically implausible RR intervals (seconds).
    Returns cleaned rr array.
    """
    rr = np.asarray(rr_seconds)
    if rr.size == 0:
        return rr
    # keep those within [min_rr, max_rr]
    mask = (rr >= min_rr) & (rr <= max_rr)
    return rr[mask]


def bpm_from_rr(rr_seconds):
    """
    Robust BPM estimate using median RR (seconds).
    """
    rr = clean_rr_intervals(rr_seconds)
    if len(rr) == 0:
        return 0.0
    median_rr = np.median(rr)
    if median_rr <= 0:
        return 0.0
    return 60.0 / median_rr



def rmssd(rr_seconds):
    """
    RMSSD in seconds (robust to outliers by cleaning RR intervals first).
    """
    rr = clean_rr_intervals(rr_seconds)
    rr = np.asarray(rr)
    if rr.size < 2:
        return 0.0
    diffs = np.diff(rr)
    ms = np.mean(diffs ** 2)
    return float(np.sqrt(ms))

def beat_regularity(rr_seconds):
    """
    Coefficient of variation of RR intervals (std/mean).
    Uses cleaned RR intervals.
    """
    rr = clean_rr_intervals(rr_seconds)
    if rr.size == 0 or np.mean(rr) == 0:
        return 0.0
    return float(np.std(rr) / (np.mean(rr) + 1e-12))

# keep the misspelled name for backward compatibility with existing imports
def beat_regulariry(rr_seconds):
    """
    Alias with the original misspelling to maintain compatibility.
    """
    return beat_regularity(rr_seconds)


def stress_level_from_rmssd(rmssd_val):
    """
    Heuristic stress level classification from RMSSD.

    These thresholds are heuristic and should be tuned with real data:
      - Low stress  : RMSSD >= 0.060 (>=60 ms)
      - Medium stress: 0.030 <= RMSSD < 0.060 (30-60 ms)
      - High stress : RMSSD < 0.030 (<30 ms)

    Parameters
    ----------
    rmssd_val : float
        RMSSD in seconds.

    Returns
    -------
    str
        'Low', 'Medium', or 'High'
    """
    # ensure float and not negative
    rv = float(rmssd_val) if rmssd_val is not None else 0.0
    if rv >= 0.06:
        return 'Low'
    elif rv >= 0.03:
        return 'Medium'
    else:
        return 'High'


def heart_health_score(bpm, rmssd_val, anomalies_count):
    """
    Compute a heart health score (0-100) based on BPM, HRV (RMSSD), and anomalies.
    """
    score = 100

    # Penalize high BPM (>120) or very low BPM (<50)
    if bpm > 120:
        score -= 20
    elif bpm < 50:
        score -= 10

    # Penalize low HRV
    if rmssd_val < 0.05:  # RMSSD < 50 ms
        score -= 20
    elif rmssd_val < 0.1:
        score -= 10

    # Penalize anomalies
    score -= anomalies_count * 5

    # Ensure score within [0, 100]
    score = max(0, min(100, score))
    return score
