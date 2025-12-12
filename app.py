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
    Cross-section of skin with microneedle patch and dielectric drug-release membrane.
    Blue block = drug reservoir under dielectric membrane.
    Blue dots = released drug molecules (only when seizure is detected).
    """
    fig, ax = plt.subplots(figsize=(4.5, 2.5))

    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, 5.0)

    # --- Skin layers ---
    ax.add_patch(Rectangle((0, 0), 10, 1.0, color="#f9e07f"))   # hypodermis / fat
    ax.add_patch(Rectangle((0, 1.0), 10, 1.8, color="#f7c0c9")) # dermis
    ax.add_patch(Rectangle((0, 2.8), 10, 0.9, color="#fdd1b0")) # epidermis
    ax.add_patch(Rectangle((0, 3.7), 10, 0.2, color="#fbe4cf")) # stratum corneum

    # --- Patch base (orange, houses dielectric membrane + electronics) ---
    patch_x, patch_w, patch_y, patch_h = 1, 8, 4.1, 0.18
    ax.add_patch(Rectangle((patch_x, patch_y), patch_w, patch_h,
                           color="#f4a259", ec="black"))

    # --- Drug reservoir (under dielectric membrane) ---
    if seizure_detected:
        res_color, res_alpha = "#93c5fd", 0.8   # partially emptied
    else:
        res_color, res_alpha = "#1d4ed8", 0.95  # full reservoir

    res_h = 0.18
    ax.add_patch(Rectangle((patch_x, patch_y - res_h), patch_w, res_h,
                           color=res_color, alpha=res_alpha, ec="black"))

    # --- Microneedles penetrating stratum corneum ---
    for x in [2.2, 5.0, 7.8]:
        ax.add_patch(Polygon([
            (x - 0.7/2, patch_y),
            (x + 0.7/2, patch_y),
            (x, patch_y - 1.2)
        ], closed=True, color="#2f4b7c", ec="black"))

    # --- Drug molecules (only when seizure detected) ---
    if seizure_detected:
        n, y_min, y_max = 60, 1.0, 3.1
        xs = np.random.uniform(patch_x + 0.3, patch_x + patch_w - 0.3, n)
        ys = np.random.uniform(y_min, y_max, n)
        ax.scatter(xs, ys, s=14, color="#3b82f6", alpha=0.85)

    # --- Simple vessels / nerves in dermis ---
    for offset in [2.0, 5.0, 8.0]:
        x_line = np.linspace(offset - 1.0, offset + 1.0, 80)
        y_line = 1.4 + 0.25 * np.sin(np.linspace(0, 4, 80))
        ax.plot(x_line, y_line, color="#d14a61", linewidth=0.9, alpha=0.8)

    # --- Labels / Annotations (dedicated space, not overlapping patch) ---
    ax.text(0.4, 4.45, "Patch Base\n(Electronics + MCU)", fontsize=7.5,
            color="#333", va="bottom")
    ax.text(0.4, 4.05, "Dielectric Drug\nReservoir", fontsize=7.5,
            color="#1d4ed8", va="bottom")
    ax.text(0.4, 2.2, "Dermis\n(Nerves / Vessels)", fontsize=7.5,
            color="#444", va="bottom")
    ax.text(0.4, 0.4, "Fat Layer", fontsize=7.5, color="#444", va="bottom")

    # Legend-style explanation below the drawing
    ax.text(5, -0.25,
            "🔵 Drug molecules released only when seizure is detected\n"
            "   (dielectric membrane activated by MCU)",
            fontsize=7, color="#3b82f6", ha="center", va="top")

    # Hide axes
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title("Smart Microneedle Patch with Dielectric Drug-Release Membrane",
                 fontsize=10, pad=6)
    plt.tight_layout()
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
            f"sEMG-based algorithm on MCU confirmed seizure.\n"
            f"Dielectric membrane activated – emergency bolus delivered.\n"
            f"IoT: Event + sEMG snapshot uploaded to cloud.\n"
            f"Please check the patient immediately."
        )
        color, border = "#ffcccc", "#ff4444"
    else:
        title = "✅ Monitoring Normal"
        body = (
            "Patient ID: EP-001\n"
            "No seizure activity detected from surface EMG.\n"
            "Patch delivering baseline dose only.\n"
            "IoT node logging electrical activity in background."
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
    st.title("Smart Microneedle Patch for Epilepsy – Interactive Demo")

    # ------------------------
    # System overview (sEMG + MCU/IoT + Dielectric Membrane)
    # ------------------------
    st.markdown("### System Overview")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            "**1. sEMG Sensing Layer**  \n"
            "- Surface EMG electrodes capture muscle/electrical activity.  \n"
            "- Signals are filtered and converted to RMS envelope.  \n"
            "- Used as the primary biomarker for imminent seizure."
        )

    with c2:
        st.markdown(
            "**2. Microcontroller + IoT Node**  \n"
            "- Runs lightweight seizure-detection algorithm on sEMG.  \n"
            "- Stores electrical activity and events on local memory/cloud.  \n"
            "- Sends trigger signal to the patch and pushes alerts to mobile."
        )

    with c3:
        st.markdown(
            "**3. Dielectric Drug-Release Membrane**  \n"
            "- Integrates with a microneedle patch + drug reservoir.  \n"
            "- When activated by MCU, electric field changes membrane properties.  \n"
            "- This opens diffusion pathways and releases anti-epileptic drug."
        )

    st.write(
        "Use the controls on the left to change signal duration and detection "
        "threshold. When a seizure is detected from sEMG, the MCU/IoT block "
        "activates the dielectric membrane, increases drug release, and sends a "
        "mobile alert to the caretaker."
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

    # Drug effect on signal
    treated_signal = apply_drug_effect(raw_signal, det_start_idx, det_end_idx)
    smoothed_treated = simple_smoothing(treated_signal, int(0.01 * fs))
    rms_treated = compute_rms(smoothed_treated, int(0.1 * fs))

    # ------------------------
    # Top status dashboard (now with MCU/IoT status)
    # ------------------------
    st.markdown("### Live Patch + IoT Status Dashboard")

    col_a, col_b, col_c, col_d = st.columns(4)

    if seizure_detected:
        seizure_text = "DETECTED"
        seizure_help = f"Seizure detected between {det_start_time:.2f}s and {det_end_time:.2f}s from sEMG."
    else:
        seizure_text = "NORMAL"
        seizure_help = "sEMG RMS did not cross threshold for the minimum duration."
    col_a.metric("🧠 Seizure Status (sEMG)", seizure_text, help=seizure_help)

    if seizure_detected:
        drug_text = "EMERGENCY BOLUS"
        drug_help = "Dielectric membrane activated; drug bolus delivered."
    else:
        drug_text = "BASELINE"
        drug_help = "Membrane closed; only basal diffusion from patch."
    col_b.metric("💊 Drug Delivery Mode", drug_text, help=drug_help)

    if seizure_detected:
        alert_text = "ALERT SENT"
        alert_help = "IoT node pushed event + notification to caretaker mobile."
    else:
        alert_text = "MONITORING"
        alert_help = "No abnormal event; system quietly logging data."
    col_c.metric("📱 Caretaker Alert", alert_text, help=alert_help)

    if seizure_detected:
        mcu_text = "TRIGGERING PATCH"
        mcu_help = "MCU drives dielectric membrane + uploads seizure data."
    else:
        mcu_text = "MONITORING"
        mcu_help = "MCU sampling sEMG & streaming logs to storage."
    col_d.metric("🧾 MCU / IoT Status", mcu_text, help=mcu_help)

    st.markdown("---")

    # ------------------------
    # Step 1 – sEMG monitoring & detection
    # ------------------------
    st.subheader("Step 1 – Surface EMG (sEMG) Monitoring & Threshold-based Seizure Detection")

    fig, axes = plt.subplots(3, 1, figsize=(8, 5.5), sharex=True)
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
    axes[2].axhline(y=threshold, linestyle="--", label="Detection Threshold")
    if seizure_detected:
        axes[2].axvspan(t[det_start_idx], t[det_end_idx], alpha=0.3, label="Detected Seizure")
    axes[2].set_title("RMS Envelope & Seizure Decision Logic")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("RMS")
    axes[2].legend()

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    st.pyplot(fig)

    if seizure_detected:
        st.success(
            f"sEMG-based algorithm detected a seizure between "
            f"**{det_start_time:.2f} s** and **{det_end_time:.2f} s**.\n\n"
            "The microcontroller activates the dielectric membrane, "
            "releasing drug and triggering an IoT alert."
        )
    else:
        st.info(
            "No seizure detected at this threshold. The microcontroller keeps the "
            "dielectric membrane closed and the patch remains in monitoring mode."
        )

    # ------------------------
    # Step 2 & 3 – Patch + Mobile Alert side by side
    # ------------------------
    st.subheader("Step 2 – Dielectric Membrane-Controlled Drug Release in the Microneedle Patch")
    st.caption("Reservoir (blue block) stores drug; droplets appear only when the MCU triggers release during seizure.")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        skin_fig = create_skin_patch_figure(seizure_detected)
        st.pyplot(skin_fig)

    with col_right:
        st.subheader("Step 3 – IoT-Connected Caretaker Mobile Alert")
        detection_info = {"start_time": det_start_time, "end_time": det_end_time}
        caretaker_phone_ui(seizure_detected, detection_info)


if __name__ == "__main__":
    main()
