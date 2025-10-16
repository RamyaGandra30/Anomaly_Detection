# batch_process_safe.py
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from preprocess import bandpass_filter, estimate_fs_from_time
from feature_extraction import detect_r_peaks, compute_rr_intervals, bpm_from_rr, rmssd, beat_regularity, stress_level_from_rmssd, heart_health_score
from predict import detect_anomalies_with_autoencoder

DATA_FOLDER = r"C:\ecg_project\data"
MODEL_PATH = r"models/autoencoder_model.h5"
MEAN_PATH = r"models/mean.npy"
STD_PATH = r"models/std.npy"
OUTPUT_FILE = r"C:\ecg_project\batch_results.xlsx"

# Load normalization stats
mean, std = np.load(MEAN_PATH), np.load(STD_PATH)

results = []
csv_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]

for filename in tqdm(csv_files):
    filepath = os.path.join(DATA_FOLDER, filename)
    anomalies_count = np.nan
    bpm = np.nan
    hrvm = np.nan
    br = np.nan
    score = np.nan
    stress = "Unknown"

    try:
        # Robust CSV reading
        try:
            df = pd.read_csv(filepath)
        except Exception:
            df = pd.read_csv(filepath, engine='python')

        # Lead selection
        lead_found = False
        for lead in ['MLII', 'V5']:
            if lead in df.columns:
                sig = df[lead].values.astype(float)
                lead_found = True
                break
        if not lead_found:
            print(f"⚠️ Failed {filename}: CSV missing MLII or V5 lead")
            results.append({
                "filename": filename,
                "anomalies": np.nan,
                "bpm": np.nan,
                "hrv_rmssd": np.nan,
                "beat_regularity": np.nan,
                "stress": stress,
                "health_score": np.nan
            })
            continue

        # Sampling frequency
        time = df['time'].values if 'time' in df.columns else None
        fs = estimate_fs_from_time(time)
        if fs is None:
            fs = 360

        # Filter
        filtered = bandpass_filter(sig, fs)

        # Feature extraction
        peaks, _ = detect_r_peaks(filtered, fs)
        rr = compute_rr_intervals(peaks, fs)
        bpm = bpm_from_rr(rr)
        hrvm = rmssd(rr)
        br = beat_regularity(rr)
        stress = stress_level_from_rmssd(hrvm)

        # Anomaly detection (safe)
        try:
            res = detect_anomalies_with_autoencoder(filtered, MODEL_PATH, mean, std, fs=fs)
            if isinstance(res, tuple) and len(res) == 2:
                positions, errors = res
                anomalies_count = len(positions)
            else:
                anomalies_count = np.nan
        except Exception as e:
            print(f"⚠️ Anomaly detection failed for {filename}: {e}")
            anomalies_count = np.nan

        # Health score
        score = heart_health_score(bpm, hrvm, anomalies_count)

    except Exception as e:
        print(f"⚠️ Failed {filename}: {e}")

    results.append({
        "filename": filename,
        "anomalies": anomalies_count,
        "bpm": bpm,
        "hrv_rmssd": hrvm,
        "beat_regularity": br,
        "stress": stress,
        "health_score": score
    })

# Save to Excel
df_results = pd.DataFrame(results)
df_results.to_excel(OUTPUT_FILE, index=False)
print(f"\n✅ Batch processing complete. Results saved to {OUTPUT_FILE}")