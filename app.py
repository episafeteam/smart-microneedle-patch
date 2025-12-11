import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go

# ------------------------
# Signal + detection logic
# ------------------------

def simulate_semg_with_seizure(duration_s=10, fs=1000):
    t = np.linspace(0, duration_s, int(duration_s * fs))
    base = 0.2 * np.random.normal(0, 1, len(t)) + 0.1 * np.sin(2 * np.pi * 10 * t)
    seizure_signal = base.copy()
    t1, t2 = 6, 7
    idx1, idx2 = int(t1 * fs), int(t2 * fs)
    seizure_signal[idx1:idx2] += 1.5 * np.random.normal(0, 1, idx2 - idx1)
    return t, seizure_signal, (idx1, idx2)


def simple_smoothing(signal, window_size):
    if window_size < 1:
        return signal
    window = np.ones(window_size) / window_size
    return np.convolve(signal, window, mode="same")


def compute_rms(signal, window_size):
    squared = np.power(signal, 2)
    window = np.ones(window_size) / window_size
    return np.sqrt(np.convolve(squared, window, mode="same"))


def detect_seizure(rms_signal, threshold, min_duration_samples):
    above = rms_signal > threshold
    if not np.any(above):
        return False, None, None
    indices = np.where(above)[0]
    splits = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
    for seg in splits:
        if len(seg) >= min_duration_samples:
            return True, int(seg[0]), int(seg[-1])
    return False, None, None


def apply_drug_effect(signal, idx_start, idx_end, reduction_factor=0.35):
    treated = signal.copy()
    if idx_start is not None and idx_end is not None:
        treated[idx_start:idx_end] *= reduction_factor
    return treated


# ------------------------
# 3D Patch visualization
# ------------------------

def create_patch_3d(seizure_detected):
    fig = go.Figure()

    # Patch: rectangle at z=0
    patch_x = [0, 2, 2, 0, 0]
    patch_y = [0, 0, 1, 1, 0]
    patch_z = [0, 0, 0, 0, 0]
    fig.add_trace(go.Scatter3d(x=patch_x, y=patch_y, z=patch_z, mode="lines", name="Microneedle Patch"))

    # Receptor layer at z=3
    rec_x = [0, 2, 2, 0, 0]
    rec_y = [0, 0, 1, 1, 0]
    rec_z = [3, 3, 3, 3, 3]
    fig.add_trace(go.Scatter3d(x=rec_x, y=rec_y, z=rec_z, mode="lines", name="Receptor Layer"))

    # Drug particles
    n_particles = 80 if seizure_detected else 10
    xs = np.random.uniform(0.1, 1.9, n_particles)
    ys = np.random.uniform(0.1, 0.9, n_particles)
    zs = np.random.uniform(0.3, 2.8 if seizure_detected else 0.8, n_particles)
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="markers", name="Drug Molecules", marker=dict(size=3)))

    fig.update_layout(
        scene=dict(
            xaxis_title="Patch Length",
            yaxis_title="Patch Width",
            zaxis_title="Depth (towards receptors)",
            aspectratio=dict(x=2, y=1, z=1.5),
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        title="3D Smart Microneedle Patch – Drug Release",
    )
    return fig


# ------------------------
# Caretaker phone UI
# ------------------------

def caretaker_phone_ui(seizure_detected, detection_info):
    if seizure_detected:
        title = "⚠️ Epileptic Seizure Detected"
        body = (
            f"Patient ID: EP-001\n"
            f"Event time: {detection_info['start_time']:.2f}–{detection_info['end_time']:.2f} s\n"
            f"Patch Status: Emergency drug bolus delivered.\n"
            f"Location: Home Monitoring\n"
            f"Please check the patient immediately."
        )
        color, border = "#ffcccc", "#ff4444"
    else:
        title = "✅ Monitoring Normal"
        body = (
            "Patient ID: EP-001\n"
            "No seizure activity detected.\n"
            "Patch Status: Baseline drug delivery.\n"
            "System: Monitoring in background."
        )
        color, border = "#ccffdd", "#22aa66"

    phone_html = f"""
    <div style="width:260px;height:480px;border-radius:30px;border:4px solid #333;
                padding:15px;background:linear-gradient(180deg,#111,#222);
                display:flex;flex-direction:column;align-items:center;color:#f5f5f5;">
        <div style="width:40%;height:8px;border-radius:10px;background:#555;margin-bottom:15px;"></div>
        <div style="width:100%;flex:1;border-radius:20px;padding:15px;background:{color};
                    color:#000;border:2px solid {border};overflow-y:auto;">
            <h4 style="margin-top:0;margin-bottom:10px;font-size:16px;">{title}</h4>
            <pre style="white-space:pre-wrap;font-size:13px;font-family:'Courier New',monospace;">{body}</pre>
        </div>
        <div style="width:40px;height:40px;border-radius:50%;background:#444;margin-top:12px;"></div>
    </div>
    """
    st.markdown(phone_html, unsafe_allow_html=True)


# ------------------------
# Streamlit App
# ------------------------

def main():
    st.set_page_config(page_title="Smart Microneedle Patch for Epilepsy", layout="wide")
    st.title("Smart Microneedle Patch for Epilepsy – Interactive Demo")

    st.write(
        "This demo shows how a smart microneedle patch monitors sEMG signals, "
        "detects seizure activity, automatically releases anti-epileptic drug, "
        "and alerts the caretaker."
    )

    # Sidebar controls
    st.sidebar.header("Control Panel")
    seed = st.sidebar.number_input("Random seed for signal", 0, 9999, 42, 1)
    np.random.seed(seed)
    duration_s = st.sidebar.slider("Signal duration (seconds)", 5, 20, 10)
    fs = 1000

    # Signal generation & processing
    t, raw_signal, (sz_start, sz_end) = simulate_semg_with_seizure(duration_s, fs)
    smoothed = simple_smoothing(raw_signal, int(0.01 * fs))
    rms = compute_rms(smoothed, int(0.1 * fs))

    # Threshold control
    base_threshold = np.mean(rms) + 2 * np.std(rms)
    st.sidebar.subheader("Seizure Detection Threshold")
    threshold_factor = st.sidebar.slider(
        "Threshold level (relative to baseline)", 0.5, 2.0, 1.0, 0.05
    )
    threshold = base_threshold * threshold_factor

    # Seizure detection
    seizure_detected, det_start_idx, det_end_idx = detect_seizure(
        rms, threshold, int(0.5 * fs)
    )
    det_start_time = t[det_start_idx] if seizure_detected else 0.0
    det_end_time = t[det_end_idx] if seizure_detected else 0.0

    # Drug effect
    treated_signal = apply_drug_effect(raw_signal, det_start_idx, det_end_idx)
    smoothed_treated = simple_smoothing(treated_signal, int(0.01 * fs))
    rms_treated = compute_rms(smoothed_treated, int(0.1 * fs))

    # ------------------------
    # Top status dashboard
    # ------------------------
    st.markdown("### Live Patch Status Dashboard")

    col_a, col_b, col_c = st.columns(3)

    # Card 1 – Seizure status
    if seizure_detected:
        seizure_text = "DETECTED"
        seizure_help = f"Seizure detected between {det_start_time:.2f}s and {det_end_time:.2f}s."
    else:
        seizure_text = "NORMAL"
        seizure_help = "RMS did not cross threshold for the minimum duration."

    col_a.metric("🧠 Seizure Status", seizure_text, help=seizure_help)

    # Card 2 – Drug delivery
    if seizure_detected:
        drug_text = "EMERGENCY BOLUS"
        drug_help = "Patch delivered a rapid dose in response to seizure."
    else:
        drug_text = "BASELINE"
        drug_help = "Patch operating in background / maintenance mode."

    col_b.metric("💊 Drug Delivery", drug_text, help=drug_help)

    # Card 3 – Alert status
    if seizure_detected:
        alert_text = "ALERT SENT"
        alert_help = "Notification pushed to registered caretaker mobile app."
    else:
        alert_text = "MONITORING"
        alert_help = "No abnormal event, no alert necessary."

    col_c.metric("📱 Alert Status", alert_text, help=alert_help)

    st.markdown("---")

    # ------------------------
    # Step 1: sEMG + RMS plots
    # ------------------------
    st.subheader("Step 1 – sEMG Monitoring & Threshold-based Seizure Detection")

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t, raw_signal)
    axes[0].set_title("Simulated sEMG Signal (Raw)")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(t, smoothed, label="Before Drug")
    axes[1].plot(t, smoothed_treated, "--", label="After Drug Release")
    axes[1].set_title("Smoothed sEMG Signal (Before vs After Drug)")
    axes[1].set_ylabel("Amplitude")
    axes[1].legend()

    axes[2].plot(t, rms, label="RMS (Before Drug)")
    axes[2].plot(t, rms_treated, "--", label="RMS (After Drug)")
    axes[2].axhline(y=threshold, linestyle="--", label="Threshold")
    if seizure_detected:
        axes[2].axvspan(t[det_start_idx], t[det_end_idx], alpha=0.3, label="Detected Seizure")
    axes[2].set_title("RMS & Seizure Detection")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("RMS")
    axes[2].legend()

    plt.tight_layout()
    st.pyplot(fig)

    if seizure_detected:
        st.success(
            f"Seizure detected between **{det_start_time:.2f} s** and "
            f"**{det_end_time:.2f} s**. Drug release is triggered."
        )
    else:
        st.info("No seizure detected at this threshold. Patch remains in monitoring mode.")

    # ------------------------
    # Step 2: 3D Patch below
    # ------------------------
    st.subheader("Step 2 – 3D Smart Microneedle Patch – Drug Release")
    fig3d = create_patch_3d(seizure_detected)
    st.plotly_chart(fig3d, use_container_width=True)

    # ------------------------
    # Step 3: Caretaker Mobile Alert
    # ------------------------
    st.subheader("Step 3 – Caretaker Mobile Alert")
    detection_info = {"start_time": det_start_time, "end_time": det_end_time}
    caretaker_phone_ui(seizure_detected, detection_info)


if __name__ == "__main__":
    main()
