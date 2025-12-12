import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.patches import Rectangle, Polygon

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
# 2D Skin + Patch visualization
# ------------------------

def create_skin_patch_figure(seizure_detected: bool):
    """
    Draw a 2D cross-section of skin with a microneedle patch.
    Blue block = drug reservoir inside the patch.
    Blue dots = drug molecules diffusing into skin.
    When seizure_detected = True -> reservoir looks partially emptied
    and many dots go deeper into the dermis.
    """
    # Smaller figure size so it doesn't dominate the page
    fig, ax = plt.subplots(figsize=(5, 3))

    # Coordinate system
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.6)

    # --- Skin layers (from bottom up) ---

    # Hypodermis / fat
    ax.add_patch(Rectangle((0, 0), 10, 1.0, color="#f9e07f"))  # yellow
    # Dermis
    ax.add_patch(Rectangle((0, 1.0), 10, 1.8, color="#f7c0c9"))  # pink
    # Epidermis
    ax.add_patch(Rectangle((0, 2.8), 10, 0.9, color="#fdd1b0"))  # light peach
    # Stratum corneum surface
    ax.add_patch(Rectangle((0, 3.7), 10, 0.2, color="#fbe4cf"))

    # --- Patch base (orange) ---

    patch_x = 1
    patch_width = 8
    patch_top_y = 4.1
    patch_height = 0.22
    ax.add_patch(
        Rectangle(
            (patch_x, patch_top_y),
            patch_width,
            patch_height,
            color="#f4a259",
            ec="black",
        )
    )

    # --- Drug reservoir (blue block) under base, above needles ---

    # Full + dark when no seizure; lighter (as if emptied) when seizure
    if seizure_detected:
        reservoir_color = "#93c5fd"   # lighter blue (partially emptied)
        reservoir_alpha = 0.8
    else:
        reservoir_color = "#1d4ed8"   # darker blue (full)
        reservoir_alpha = 0.95

    reservoir_height = 0.20
    reservoir_bottom_y = patch_top_y - reservoir_height
    ax.add_patch(
        Rectangle(
            (patch_x, reservoir_bottom_y),
            patch_width,
            reservoir_height,
            color=reservoir_color,
            alpha=reservoir_alpha,
            ec="black",
        )
    )

    # --- Microneedles (triangles) ---

    needle_positions = [2.2, 5.0, 7.8]
    for x_center in needle_positions:
        width = 0.8
        height = 1.4
        top_y = patch_top_y
        points = [
            (x_center - width / 2, top_y),
            (x_center + width / 2, top_y),
            (x_center, top_y - height),
        ]
        ax.add_patch(Polygon(points, closed=True, color="#2f4b7c", ec="black"))

    # --- Drug molecules (blue dots) ---

    if seizure_detected:
        # Many droplets, going deeper into dermis
        n_dots = 80
        y_min, y_max = 1.0, 3.2   # dermis + lower epidermis
    else:
        # Very few droplets, staying close to surface (maintenance)
        n_dots = 12
        y_min, y_max = 3.1, 3.7   # upper epidermis only

    xs = np.random.uniform(patch_x + 0.3, patch_x + patch_width - 0.3, n_dots)
    ys = np.random.uniform(y_min, y_max, n_dots)
    ax.scatter(xs, ys, s=18, color="#3b82f6", alpha=0.85, label="Drug molecules")

    # --- Simple nerve / vessel hints in dermis ---

    for offset in [2.0, 5.0, 8.0]:
        x_line = np.linspace(offset - 1.2, offset + 1.2, 100)
        y_line = 1.5 + 0.25 * np.sin(np.linspace(0, 4, 100))
        ax.plot(x_line, y_line, color="#d14a61", linewidth=1.1, alpha=0.8)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Cross-section of Skin with Smart Microneedle Patch")
    ax.legend(loc="upper right")

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

    if seizure_detected:
        seizure_text = "DETECTED"
        seizure_help = f"Seizure detected between {det_start_time:.2f}s and {det_end_time:.2f}s."
    else:
        seizure_text = "NORMAL"
        seizure_help = "RMS did not cross threshold for the minimum duration."
    col_a.metric("🧠 Seizure Status", seizure_text, help=seizure_help)

    if seizure_detected:
        drug_text = "EMERGENCY BOLUS"
        drug_help = "Patch delivered a rapid dose in response to seizure."
    else:
        drug_text = "BASELINE"
        drug_help = "Patch operating in background / maintenance mode."
    col_b.metric("💊 Drug Delivery", drug_text, help=drug_help)

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
    # Step 2: Skin + Patch cartoon
    # ------------------------
    st.subheader("Step 2 – Drug Release into Skin via Smart Microneedle Patch")
    st.caption("Blue droplets represent drug molecules diffusing into skin layers.")
    skin_fig = create_skin_patch_figure(seizure_detected)
    st.pyplot(skin_fig)

    # ------------------------
    # Step 3: Caretaker Mobile Alert
    # ------------------------
    st.subheader("Step 3 – Caretaker Mobile Alert")
    detection_info = {"start_time": det_start_time, "end_time": det_end_time}
    caretaker_phone_ui(seizure_detected, detection_info)


if __name__ == "__main__":
    main()

