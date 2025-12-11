import numpy as np
import matplotlib.pyplot as plt

from seizure_detection import simple_smoothing, compute_rms, detect_seizure

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

def main():
    fs = 1000  # sampling frequency (Hz)
    t, raw_signal, (sz_start_idx, sz_end_idx) = simulate_semg_with_seizure()

    # Simple smoothing instead of full filter
    smoothed = simple_smoothing(raw_signal, window_size=int(0.01 * fs))

    # Compute RMS with 0.1 s window
    window_size = int(0.1 * fs)
    rms = compute_rms(smoothed, window_size)

    # Threshold based on stats
    threshold = np.mean(rms) + 2 * np.std(rms)
    min_duration_samples = int(0.5 * fs)  # 0.5 s

    detected, start_idx, end_idx = detect_seizure(
        rms, threshold, min_duration_samples
    )

    print("=== Smart Microneedle Patch Simulation ===")
    print(f"Sampling rate (Hz): {fs}")
    print(f"RMS threshold: {threshold:.3f}")

    if detected:
        start_time = t[start_idx]
        end_time = t[end_idx]
        print(f"⚠️ Seizure detected between {start_time:.2f}s and {end_time:.2f}s")
        print("👉 Command: TRIGGER_DRUG_RELEASE")
    else:
        print("✅ No seizure detected.")
        print("👉 Command: NORMAL_MONITORING")

    # Plot for demo
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(t, raw_signal)
    axes[0].set_title("Simulated sEMG Signal (Raw)")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(t, smoothed)
    axes[1].set_title("Smoothed sEMG Signal")
    axes[1].set_ylabel("Amplitude")

    axes[2].plot(t, rms, label="RMS")
    axes[2].axhline(y=threshold, linestyle="--", label="Threshold")
    if detected:
        axes[2].axvspan(t[start_idx], t[end_idx], alpha=0.3, label="Detected Seizure")
    axes[2].set_title("RMS & Seizure Detection")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("RMS")
    axes[2].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
