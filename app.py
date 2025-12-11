import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go

# ------------------------
# Signal + detection logic
# ------------------------

def simulate_semg_with_seizure(duration_s=10, fs=1000):
    """
    Simulate an sEMG-like signal with a seizure event.
    """
    t = np.linspace(0, duration_s, int(duration_s * fs))

    # Base muscle activity: random noise + small oscillations
    base = 0.2 * np.random.normal(0, 1, len(t)) + 0.1 * np.sin(2 * np.pi * 10 * t)

    # Copy and add a seizure-like burst between t1 and t2
    seizure_signal = base.copy()
    t1, t2 = 6, 7  # seconds
    idx1, idx2 = int(t1 * fs), int(t2 * fs)
    seizure_signal[idx1:idx2] += 1.5 * np.random.normal(0, 1, idx2 - idx1)

    return t, seizure_signal, (idx1, idx2)


def simple_smoothing(signal, window_size):
    """
    Simple moving-average smoothing to mimic basic filtering.
    """
    if window_size < 1:
        return signal
    window = np.ones(window_size) / window_size
    smoothed = np.convolve(signal, window, mode="same")
    return smoothed


def compute_rms(signal, window_size):
    """
    Compute moving RMS of the signal.
    window_size: number of samples in each window.
    """
    squared = np.power(signal, 2)
    window = np.ones(window_size) / window_size
    rms = np.sqrt(np.convolve(squared, window, mode="same"))
    return rms


def detect_seizure(rms_signal, threshold, min_duration_samples):
    """
    Detect seizure if RMS stays above threshold for
    at least min_duration_samples.
    Returns:
        detected (bool),
        start_index (int or None),
        end_index (int or None)
    """
    above = rms_signal > threshold
    if not np.any(above):
        return False, None, None

    indices = np.where(above)[0]
    splits = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)

    for seg in splits:
        if len(seg) >= min_duration_samples:
            return True, int(seg[0]), int(seg[-1])

    return False, None, None


def apply_drug_effect(signal, idx_start, idx_end, reduction_factor=0.3):
    """
    Reduce the amplitude of the signal in the seizure region
    to simulate drug action.
    """
    treated = signal.copy()
    if idx_start is not None and idx_end is not None:
        treated[idx_start:idx_end] *= reduction_factor
    return treated


# ------------------------
# 3D Patch visualization
# ------------------------

def create_patch_3d(seizure_detected):
    """
    Create a simple 3D visualization:
    - A rectangular microneedle patch
    - Drug molecules travelling towards receptors
    - Receptor layer
    """
    fig = go.Figure()

    # Patch: a rectangle in XY plane at z = 0
    patch_x = [0, 2, 2, 0, 0]
    patch_y = [0, 0, 1, 1, 0]
    patch_z = [0, 0, 0, 0, 0]
    fig.add_trace(
        go.Scatter3d(
            x=patch_x,
            y=patch_y,
            z=patch_z,
            mode="lines",
            name="Microneedle Patch",
        )
    )

    # Receptor layer: a plane at z = 3
    rec_x = [0, 2, 2, 0, 0]
    rec_y = [0, 0, 1, 1, 0]
    rec_z = [3, 3, 3, 3, 3]
    fig.add_trace(
        go.Scatter3d(
            x=rec_x,
            y=rec_y,
            z=rec_z,
            mode="lines",
            name="Receptor Layer",
        )
    )

    # Drug particles
    if seizure_detected:
        n_particles = 80
    else:
        n_particles = 10  # minimal background diffusion

    # Start near patch (z ~ 0.2), move toward receptors (z ~ 3)
    xs = np.random.uniform(0.1, 1.9, n_particles)
    ys = np.random.uniform(0.1, 0.9, n_particles)
    if seizure_detected:
        zs = np.random.uniform(0.3, 2.8, n_particles)
    else:
        zs = np.random.uniform(0.3, 0.8, n_particles)

    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="markers",
            name="Drug Molecules",
            marker=dict(size=3),
        )
    )

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
# Caretaker phone UI block
# ------------------------

def caretaker_phone_ui(seizure_detected, detection_info):
    """
    Render a simple 'smartphone' view with alert message.
    """
    if seizure_detected:
        title = "⚠️ Epileptic Seizure Detected"
        body = (
            f"Patient ID: EP-001\n"
            f"Event time: {detection_info['start_time']:.2f}–{detection_info['end_time']:.2f} s\n"
            f"Patch Status: Emergency drug bolus delivered.\n"
            f"Location: Home Monitoring\n"
            f"Please check the patient immediately."
        )
        color = "#ffcccc"
        border = "#ff4444"
    else:
        title = "✅ Monitoring Normal"
        body = (
            "Patient ID: EP-001\n"
            "No seizure activity detected.\n"
            "Patch Status: Baseline drug delivery.\n"
            "System: Monitoring in background."
        )
        color = "#ccffdd"
        border = "#22aa66"

    phone_html = f"""
    <div style="
        width: 260px;
        height: 480px;
        border-radius: 30px;
        border: 4px solid #333;
        padding: 15px;
        background: linear-gradient(180deg, #111, #222);
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        align-items: center;
        color: #f5f5f5;
        font-family: Arial, sans-serif;
        ">
        <div style="
            width: 40%;
            height: 8px;
            border-radius: 10px;
            background: #555;
            margin-bottom: 15px;">
        </div>
        <div style="
            width: 100%;
            flex: 1;
            border-radius: 20px;
            padding: 15px;
            background: {color};
            color: #000;
            border: 2px solid {border};
            box-sizing: border-box;
            overflow-y: auto;
            ">
            <h4 style="margin-top: 0; margin-bottom: 10px; font-size: 16px;">{title}</h4>
            <pre style="white-space: pre-wrap; font-size: 13px; font-family: 'Courier New', monospace;">{body}</pre>
        </div>
        <div style="
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background: #444;
            margin-top: 12px;">
        </div>
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
        "This web demo shows how a smart microneedle patch monitors sEMG signals, "
        "detects seizure activity, automatically releases anti-epileptic drug, "
        "and alerts the caretaker."
    )

    # Sidebar controls
    st.sidebar.header("Control Panel")

    # Seed for reproducibility
    seed = st.sidebar.number_input("Random seed for signal", min_value=0, max_value=9999, value=42, step=1)
    np.random.seed(seed)

    # Simulation parameters
    duration_s = st.sidebar.slider("Signal duration (seconds)", 5, 20, 10)
    fs = 1000

    # Generate signal
    t, raw_signal, (sz_start_idx_true, sz_end_idx_true) = simulate_semg_with_seizure(duration_s, fs)

    # Smoothing and RMS
    smooth_window = int(0.01 * fs)
    smoothed = simple_smoothing(raw_signal, smooth_window)

    rms_window = int(0.1 * fs)
    rms = compute_rms(smoothed, rms_window)

    # Base threshold from stats
    base_threshold = np.mean(rms) + 2 * np.std(rms)

    st.sidebar.subheader("Seizure Detection Threshold")
    threshold_factor = st.sidebar.slider(
        "Threshold level (relative to baseline)",
        min_value=0.5,
        max_value=2.0,
        step=0.05,
        value=1.0,
    )
    threshold = base_threshold * threshold_factor

    min_duration_samples = int(0.5 * fs)

    seizure_detected, det_start_idx, det_end_idx = detect_seizure(
        rms, threshold, min_duration_samples
    )

    if seizure_detected:
        det_start_time = t[det_start_]()
