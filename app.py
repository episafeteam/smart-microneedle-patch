import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.patches import Rectangle, Polygon
import datetime
import time

# ------------------------
# Signal + detection logic
# ------------------------

def simulate_semg_with_seizure(duration_s=10, fs=1000):
    """Simulate surface EMG in realistic µV range."""
    t = np.linspace(0, duration_s, int(duration_s * fs))

    base_shape = 0.5 * np.random.normal(0, 1, len(t)) + 0.25 * np.sin(2 * np.pi * 10 * t)
    seizure_shape = base_shape.copy()
    t1, t2 = 6, 7
    idx1, idx2 = int(t1 * fs), int(t2 * fs)
    seizure_shape[idx1:idx2] += 3.0 * np.random.normal(0, 1, idx2 - idx1)

    baseline_scale = 50e-6   # 50 µV
    seizure_scale = 300e-6   # ~300 µV spikes

    signal = baseline_scale * base_shape
    signal[idx1:idx2] += seizure_scale * np.random.normal(0, 1, idx2 - idx1)

    return t, signal, (idx1, idx2)


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
# Skin + Patch visualization
# ------------------------

def create_skin_patch_figure(seizure_detected: bool):
    """Compact skin cross-section with dielectric drug-release membrane."""
    fig, ax = plt.subplots(figsize=(4.5, 2.5))

    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, 5.0)

    # Skin layers
    ax.add_patch(Rectangle((0, 0), 10, 1.0, color="#f9e07f"))
    ax.add_patch(Rectangle((0, 1.0), 10, 1.8, color="#f7c0c9"))
    ax.add_patch(Rectangle((0, 2.8), 10, 0.9, color="#fdd1b0"))
    ax.add_patch(Rectangle((0, 3.7), 10, 0.2, color="#fbe4cf"))

    # Patch base
    patch_x, patch_w, patch_y, patch_h = 1, 8, 4.1, 0.18
    ax.add_patch(Rectangle((patch_x, patch_y), patch_w, patch_h, color="#f4a259", ec="black"))

    # Reservoir (blue)
    res_color, res_alpha = ("#93c5fd", 0.8) if seizure_detected else ("#1d4ed8", 0.95)
    ax.add_patch(Rectangle((patch_x, patch_y - 0.18), patch_w, 0.18, color=res_color, alpha=res_alpha, ec="black"))

    # Needles
    for x in [2.2, 5.0, 7.8]:
        ax.add_patch(Polygon([
            (x - 0.35, patch_y),
            (x + 0.35, patch_y),
            (x, patch_y - 1.2)
        ], closed=True, color="#2f4b7c", ec="black"))

    # Drug molecules only when seizure
    if seizure_detected:
        xs = np.random.uniform(patch_x + 0.3, patch_x + patch_w - 0.3, 60)
        ys = np.random.uniform(1.0, 3.1, 60)
        ax.scatter(xs, ys, s=14, color="#3b82f6", alpha=0.85)

    # Labels
    ax.text(0.4, 4.45, "Patch Base\n(MCU + Membrane)", fontsize=7.5, color="#333", va="bottom")
    ax.text(0.4, 4.05, "Drug Reservoir", fontsize=7.5, color="#1d4ed8", va="bottom")
    ax.text(0.4, 2.2, "Dermis\n(Nerves / Vessels)", fontsize=7.5, color="#444", va="bottom")
    ax.text(0.4, 0.4, "Fat Layer", fontsize=7.5, color="#444", va="bottom")
    ax.text(5, -0.25, "🔵 Drug molecules released only when seizure is detected", fontsize=7, color="#3b82f6", ha="center")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Smart Microneedle Patch (Cross-section)", fontsize=10)
    plt.tight_layout()
    return fig


# ------------------------
# IoT Live Log Generator
# ------------------------

def generate_iot_log(seizure_detected):
    """Simulate real-time IoT telemetry from MCU."""
    log_entries = []
    now = datetime.datetime.now()
    for i in range(10):
        timestamp = (now + datetime.timedelta(seconds=i)).strftime("%H:%M:%S")
        voltage = np.random.uniform(45, 60) if not seizure_detected else np.random.uniform(120, 450)
        status = "Normal" if voltage < 100 else "Seizure Spike"
        log_entries.append(f"{timestamp} | {voltage:6.1f} µV | {status}")
    return "\n".join(log_entries)


# ------------------------
# Caretaker phone UI
# ------------------------

def caretaker_phone_ui(seizure_detected, detection_info):
    if seizure_detected:
        title = "⚠️ Epileptic Seizure Detected"
        body = (
            f"Patient ID: EP-001\n"
            f"Event: {detection_info['start_time']:.2f}–{detection_info['end_time']:.2f} s\n"
            f"sEMG spike detected (100–500 µV range).\n"
            f"MCU activated dielectric membrane → drug release.\n"
            f"IoT uploaded data to cloud.\n"
            f"Immediate caretaker attention required."
        )
        color, border = "#ffcccc", "#ff4444"
    else:
        title = "✅ Monitoring Normal"
        body = (
            "Patient ID: EP-001\n"
            "Surface EMG stable (~50 µV baseline).\n"
            "No seizure detected.\n"
            "MCU logging data, membrane inactive."
        )
        color, border = "#ccffdd", "#22aa66"

    phone_html = f"""
    <div style="width:230px;height:420px;border-radius:30px;border:4px solid #333;
                padding:12px;background:linear-gradient(180deg,#111,#222);
                display:flex;flex-direction:column;align-items:center;color:#f5f5f5;">
        <div style="width:40%;height:8px;border-radius:10px;background:#555;margin-bottom:12px;"></div>
        <div style="width:100%;flex:1;border-radius:20px;padding:12px;background:{color};
                    color:#000;border:2px solid {border};overflow-y:auto;">
            <h4 style="margin-top:0;margin-bottom:8px;font-size:14px;">{title}</h4>
            <pre style="white-space:pre-wrap;font-size:11px;font-family:'Courier New',monospace;">{body}</pre>
        </div>
        <div style="width:36px;height:36px;border-radius:50%;background:#444;margin-top:10px;"></div>
    </div>
    """
    st.markdown(phone_html, unsafe_allow_html=True)


# ------------------------
# Streamlit App
# ------------------------

def main():
    st.set_page_config(page_title="Smart Microneedle Patch for Epilepsy", layout="wide")
    st.title("Smart Microneedle Patch for Epilepsy – Integrated IoT Demo")

    # Hardware system diagram (static)
    st.image(
        "https://raw.githubusercontent.com/yeshwanthya/smart-patch-hardware-diagram/main/system_diagram.png",
        caption="System overview: sEMG electrodes → MCU/IoT → Dielectric Patch → Caretaker Alert",
        use_column_width=True
    )

    st.markdown(
        "This system uses **surface EMG sensors** to detect electrical activity (50–500 µV). "
        "A **microcontroller with IoT** connectivity processes the data and, if seizure activity "
        "is detected, triggers a **dielectric membrane** in the microneedle patch to release "
        "drug molecules and simultaneously sends an alert to the caretaker’s mobile device."
    )

    # Sidebar
    st.sidebar.header("Control Panel")
    seed = st.sidebar.number_input("Random seed", 0, 9999, 42, 1)
    np.random.seed(seed)
    duration_s = st.sidebar.slider("Signal duration (seconds)", 5, 20, 10)
    fs = 1000

    # Simulate signal
    t, raw_signal, (sz_start, sz_end) = simulate_semg_with_seizure(duration_s, fs)
    smoothed = simple_smoothing(raw_signal, int(0.01 * fs))
    rms = compute_rms(smoothed, int(0.1 * fs))

    base_threshold = np.mean(rms) + 2 * np.std(rms)
    threshold_factor = st.sidebar.slider("Detection threshold multiplier", 0.5, 2.0, 1.0, 0.05)
    threshold = base_threshold * threshold_factor

    seizure_detected, det_start_idx, det_end_idx = detect_seizure(rms, threshold, int(0.5 * fs))
    det_start_time = t[det_start_idx] if seizure_detected else 0.0
    det_end_time = t[det_end_idx] if seizure_detected else 0.0

    treated_signal = apply_drug_effect(raw_signal, det_start_idx, det_end_idx)
    smoothed_treated = simple_smoothing(treated_signal, int(0.01 * fs))
    rms_treated = compute_rms(smoothed_treated, int(0.1 * fs))

    # Dashboard
    st.markdown("### Live System Status")
    cols = st.columns(4)
    cols[0].metric("🧠 Seizure Status", "Detected" if seizure_detected else "Normal")
    cols[1].metric("💊 Drug Delivery", "Active" if seizure_detected else "Baseline")
    cols[2].metric("📡 IoT Upload", "In Progress" if seizure_detected else "Logging")
    cols[3].metric("⚙️ MCU Status", "Triggering Patch" if seizure_detected else "Monitoring")

    st.markdown("---")

    # Step 1 – sEMG
    st.subheader("Step 1 – Surface EMG (sEMG) Monitoring")
    fig, axes = plt.subplots(3, 1, figsize=(8, 5.5), sharex=True)
    axes[0].plot(t, raw_signal * 1e6)
    axes[0].set_title("Surface EMG Signal")
    axes[0].set_ylabel("Amplitude (µV)")
    axes[1].plot(t, smoothed * 1e6, label="Before Drug")
    axes[1].plot(t, smoothed_treated * 1e6, "--", label="After Drug Release")
    axes[1].legend()
    axes[1].set_ylabel("Amplitude (µV)")
    axes[2].plot(t, rms * 1e6, label="RMS (Before Drug)")
    axes[2].plot(t, rms_treated * 1e6, "--", label="RMS (After Drug)")
    axes[2].axhline(y=threshold * 1e6, linestyle="--", label="Threshold")
    axes[2].legend()
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("RMS (µV)")
    plt.tight_layout()
    st.pyplot(fig)

    # Step 2 + 3 – Patch + Mobile
    st.subheader("Step 2 – Patch Drug Release and IoT Alert")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.pyplot(create_skin_patch_figure(seizure_detected))
    with col2:
        caretaker_phone_ui(seizure_detected, {"start_time": det_start_time, "end_time": det_end_time})

    # Step 4 – Live IoT Log Panel
    st.subheader("Step 4 – Live IoT Telemetry Log")
    st.caption("Simulated data stream from MCU/IoT node showing surface EMG readings in microvolts.")
    st.text(generate_iot_log(seizure_detected))


if __name__ == "__main__":
    main()
