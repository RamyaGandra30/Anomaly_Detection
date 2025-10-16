# dashboard/app.py (FULL FIXED VERSION)
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
from dotenv import load_dotenv
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocess import bandpass_filter, estimate_fs_from_time
from src.feature_extraction import detect_r_peaks, compute_rr_intervals, bpm_from_rr, rmssd, beat_regularity
from src.predict import detect_anomalies_with_autoencoder
from alerts.email_alerts import send_email_alert

load_dotenv()

st.set_page_config(page_title='ECG Health Monitor (Balanced)', layout='wide')
st.title("ECG Health Monitor — Balanced Mode (Improved Accuracy)")

# Sidebar controls
st.sidebar.header("Detection settings")
sensitivity = st.sidebar.slider("Anomaly sensitivity (lower = more sensitive)", 0.5, 6.0, 3.0, 0.1)
inject_spike = st.sidebar.checkbox("Inject test spike (for verification)", False)
auto_send = st.sidebar.checkbox("Auto-send email on Critical (once per session)", True)
show_debug = st.sidebar.checkbox("Show debug info", False)

if 'alert_sent' not in st.session_state:
    st.session_state['alert_sent'] = False

# File upload
uploaded = st.file_uploader("Upload ECG CSV (prefer MLII/V5)", type=['csv'])
if uploaded is None:
    st.info("Upload a CSV with `time` and `MLII`/`V5` or any numeric ECG column.")
    st.stop()

df = pd.read_csv(uploaded)

# Choose lead
preferred = ['MLII', 'V5']
sig = None
lead_used = None
for p in preferred:
    if p in df.columns:
        sig = df[p].values.astype(float)
        lead_used = p
        break
if sig is None:
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) == 0:
        st.error("No numeric columns found in CSV.")
        st.stop()
    lead_used = numeric_cols[0]
    sig = df[lead_used].values.astype(float)
    st.warning(f"Using numeric column: {lead_used}")

st.subheader(f"Using lead: {lead_used}")

# Estimate sampling frequency
fs = estimate_fs_from_time(df['time'].values) if 'time' in df.columns else None
if fs is None:
    fs = 360
    st.info(f"No time column detected — defaulting fs={fs} Hz")

# Preprocess
filtered = bandpass_filter(sig, fs)

# Preview plot
preview_s = 10
end_idx = min(len(filtered), int(preview_s * fs))
t = np.arange(len(filtered)) / fs
fig, ax = plt.subplots(figsize=(10,3))
ax.plot(t[:end_idx], sig[:end_idx], alpha=0.35, label='Raw')
ax.plot(t[:end_idx], filtered[:end_idx], label='Filtered')
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.legend()
st.pyplot(fig)

# Feature extraction
peaks, props = detect_r_peaks(filtered, fs)
rr = compute_rr_intervals(peaks, fs)
bpm = bpm_from_rr(rr)
hrv_seconds = rmssd(rr)  # seconds
rr_reg = beat_regularity(rr)

# anomaly detection
model_path = os.path.join('models', 'autoencoder_model.h5')
anomaly_positions = []
anomaly_severity = []
detection_info = None

if os.path.exists(model_path):
    mean_path = os.path.join('models', 'mean.npy')
    std_path = os.path.join('models', 'std.npy')
    if not (os.path.exists(mean_path) and os.path.exists(std_path)):
        st.error("Normalization stats missing (models/mean.npy or std.npy).")
    else:
        mean = np.load(mean_path, allow_pickle=True)
        std = np.load(std_path, allow_pickle=True)
        try:
            detection_info = detect_anomalies_with_autoencoder(
                filtered, model_path, mean, std,
                fs=fs, std_multiplier=float(sensitivity),
                min_consecutive_windows=2,
                inject_test_spike=inject_spike
            )
            anomaly_positions = detection_info.get("anomaly_positions", [])
            anomaly_errors = detection_info.get("anomaly_errors", [])
            if detection_info is not None and 'errors' in detection_info:
                errors = np.array(detection_info['errors'])
                if len(errors) > 0:
                    max_err = np.max(errors)
                    confidence_scores = 100 - (errors / max_err * 100)  # 0-100 scale
                    mean_confidence = np.mean(confidence_scores)
                else:
                    mean_confidence = 100.0  # No error → full confidence
            else:
                mean_confidence = 100.0

            # Numeric severity (0=Low,1=Moderate,2=High)
            threshold = detection_info.get("threshold", 0.0)
            for e in anomaly_errors:
                if e > threshold * 2:
                    anomaly_severity.append(2)
                elif e > threshold:
                    anomaly_severity.append(1)
                else:
                    anomaly_severity.append(0)

        except Exception as e:
            st.error(f"Autoencoder detection failed: {e}")
else:
    st.warning("Autoencoder model not found. Anomaly detection disabled.")

# Heart health score
def heart_health_score(bpm, hrvm, rr_regularity, anomalies_severity):
    bpm_penalty = (50 - bpm) * 0.5 if bpm < 50 else (bpm - 100) * 0.5 if bpm > 100 else 0
    rmssd_penalty = max(0, (0.03 - hrvm) * 200)
    rr_penalty = rr_regularity * 50
    anomaly_penalty = sum(anomalies_severity) * 5  # scaled
    score = 100 - (bpm_penalty + rmssd_penalty + rr_penalty + anomaly_penalty)
    return max(0, min(100, round(score, 1)))

# Stress estimation
def stress_level_from_features(hrvm, rr_regularity, anomalies_severity):
    score = hrvm*100 - rr_regularity*50 + sum(anomalies_severity)*5
    if score > 10:
        return "Low"
    elif score > -10:
        return "Medium"
    else:
        return "High"

score = heart_health_score(bpm, hrv_seconds, rr_reg, anomaly_severity)
stress = stress_level_from_features(hrv_seconds, rr_reg, anomaly_severity)

# Severity mapping
if score < 40 or sum(anomaly_severity) > 3:
    severity = "Critical"
elif score < 60 or sum(anomaly_severity) > 1.5:
    severity = "High"
elif score < 80:
    severity = "Moderate"
else:
    severity = "Normal"

# Display metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric('BPM', round(bpm, 1))
col2.metric('Anomalies', len(anomaly_positions))
col3.metric('HRV (RMSSD)', round(hrv_seconds, 4))
col4.metric('Heart Health Score', score)


st.write('🫀 Beat Regularity (std/mean RR):', round(rr_reg, 4))
st.write('😥 Estimated Stress Level:', stress)

# Debug info
if show_debug and detection_info is not None:
    st.markdown("### Debug info")
    st.write(f"Window size: {detection_info.get('window_size',0)}, step: {detection_info.get('step',0)}")
    st.write(f"Number of windows: {detection_info.get('n_windows',0)}")
    st.write(f"Threshold (median+mult*MAD): {detection_info.get('threshold',0.0)}")
    st.write("Errors (first 20):", np.array(detection_info.get('errors',[]))[:20])

# Plot anomalies
fig2, ax2 = plt.subplots(figsize=(10,3))
ax2.plot(t[:end_idx], filtered[:end_idx], label='Filtered')
pk_times = peaks / fs
ax2.scatter(pk_times[pk_times < t[end_idx-1]], filtered[peaks[pk_times < t[end_idx-1]]], s=10, color='black', label='R-peaks')
for i, pos in enumerate(anomaly_positions):
    if pos < end_idx:
        alpha_val = 0.2 + 0.3 * anomaly_severity[i]
        ax2.axvspan((pos-20)/fs, (pos+20)/fs, color='red', alpha=alpha_val)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Amplitude")
ax2.legend()
st.pyplot(fig2)

# Anomaly table
if len(anomaly_positions) > 0:
    st.markdown("### Detected anomaly events (model)")
    rows = []
    for i, pos in enumerate(anomaly_positions):
        recon_error = round(detection_info.get('anomaly_errors',[0])[i],6) if i < len(detection_info.get('anomaly_errors',[])) else None
        sev_val = ["Low","Moderate","High"][anomaly_severity[i]] if i < len(anomaly_severity) else "Unknown"
        rows.append({
            "index": i+1,
            "sample": pos,
            "time_s": round(pos/float(fs), 3),
            "recon_error": recon_error,
            "severity": sev_val
        })
    st.table(pd.DataFrame(rows))
else:
    st.info("No model anomalies detected.")

# Auto-send email if critical
DEFAULT_EMAIL = "22eg105h66@anurag.edu.in"
recipient_email = st.text_input("Enter email for alert (default provided)", value=DEFAULT_EMAIL)

if auto_send and severity in ["Critical","High"] and not st.session_state['alert_sent']:
    to_email = recipient_email.strip() if recipient_email.strip() != "" else DEFAULT_EMAIL
    try:
        send_email_alert(
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            username=os.getenv("EMAIL_USER"),
            password=os.getenv("EMAIL_PASS"),
            to_email=to_email,
            subject=f"🚨 ECG Alert — {severity}",
            body=(f"ECG Health Alert\nLead: {lead_used}\nTime: {datetime.utcnow().isoformat()} UTC\n"
                  f"BPM: {round(bpm,1)}\nHRV (ms): {round(hrv_seconds,1)}\nAnomalies: {len(anomaly_positions)}\nScore: {score}\nSeverity: {severity}\n")
        )
        st.success(f"✅ Alert email automatically sent to {to_email}")
        st.session_state['alert_sent'] = True
    except Exception as e:
        st.error(f"❌ Failed to send automatic alert email: {e}")

# Manual email send
if st.button("Send Alert Email (manual)"):
    to_email = recipient_email.strip() if recipient_email.strip() != "" else DEFAULT_EMAIL
    try:
        send_email_alert(
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            username=os.getenv("EMAIL_USER"),
            password=os.getenv("EMAIL_PASS"),
            to_email=to_email,
            subject=f"ECG Alert — {severity}",
            body=(f"ECG Health Alert\nLead: {lead_used}\nTime: {datetime.utcnow().isoformat()} UTC\n"
                  f"BPM: {round(bpm,1)}\nHRV (ms): {round(hrv_seconds,1)}\nAnomalies: {len(anomaly_positions)}\nScore: {score}\nSeverity: {severity}\n")
        )
        st.success(f"✅ Alert email sent to {to_email}")
    except Exception as e:
        st.error(f"❌ Failed to send email: {e}")
