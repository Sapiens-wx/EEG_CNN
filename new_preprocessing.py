# new_preprocessing.py
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

def preprocess_eeg_csv(file_path, window_size=256, sliding_step=128, lowcut=5, highcut=40, fs=256):
    """
    Standard Muse EEG preprocessing:
    - Subtract AUX (Right AUX) from all channels
    - Apply bandpass filter (default: 5–40Hz)
    - Segment data using sliding window

    Args:
        file_path (str): Path to the EEG CSV file (must include TP9, AF7, AF8, TP10, Right AUX)
        window_size (int): Number of samples per window
        sliding_step (int): Step size between windows (controls overlap)
        lowcut (float): Lower bound of bandpass filter (Hz)
        highcut (float): Upper bound of bandpass filter (Hz)
        fs (int): Sampling frequency (default: 256Hz)

    Returns:
        np.ndarray: EEG windows of shape (num_windows, window_size, 4)
    """
    df = pd.read_csv(file_path)
    channels = ["TP9", "AF7", "AF8", "TP10", "Right AUX"]
    for ch in channels:
        if ch not in df.columns:
            raise ValueError(f"Missing EEG channel: {ch}")

    # remove aux
    signals = df[["TP9", "AF7", "AF8", "TP10"]].copy()
    for ch in signals.columns:
        signals[ch] = signals[ch] - df["Right AUX"]

    # Apply bandpass filter
    nyq = 0.5 * fs
    b, a = butter(4, [lowcut / nyq, highcut / nyq], btype='band')
    for ch in signals.columns:
        signals[ch] = filtfilt(b, a, signals[ch].values)

    # Segment data into overlapping windows
    arr = signals.values
    windows = []
    for start in range(0, len(arr) - window_size + 1, sliding_step):
        windows.append(arr[start:start + window_size])

    return np.array(windows)
