import numpy as np

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
    window_size: number of samples in each window
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
import numpy as np

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
    window_size: number of samples in each window
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
