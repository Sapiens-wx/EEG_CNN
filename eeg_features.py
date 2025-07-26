import numpy as np
import pywt
from scipy.signal import spectrogram

def extract_sfft(segment, fs=256, nperseg=64, noverlap=32, target_shape=(33, 5)):
    """
    Extract short-time Fourier transform (SFFT) features from a single EEG segment.

    Args:
        segment (np.ndarray): Shape (256, 4), raw EEG segment (time, channels)
        fs (int): Sampling frequency (default 256 Hz)
        nperseg (int): Length of each FFT segment (default 64 samples ≈ 0.25s)
        noverlap (int): Overlap between segments (default 32 = 50%)

    Returns:
        np.ndarray: Shape (4, freq_bins, time_bins), spectrogram magnitude for each channel
    """
    assert segment.shape == (256, 4), f"Expected segment shape (256, 4), got {segment.shape}"
    result = []

    for ch in range(4):
        f, t, Sxx = spectrogram(segment[:, ch], fs=fs, nperseg=nperseg, noverlap=noverlap)
        Sxx = Sxx[:target_shape[0] * target_shape[1]]  # flatten then reshape
        Sxx = Sxx.flatten()
        if len(Sxx) < target_shape[0] * target_shape[1]:
            pad = np.zeros(target_shape[0] * target_shape[1] - len(Sxx))
            Sxx = np.concatenate([Sxx, pad])
        Sxx = Sxx[:target_shape[0] * target_shape[1]].reshape(target_shape)
        result.append(Sxx)

    return np.array(result)  # shape: (4, target_shape[0], target_shape[1])



def extract_dwt(segment, wavelet='db4', level=3, target_shape=(33, 5)):
    """
    Extract DWT features from a single EEG segment.

    Args:
        segment (np.ndarray): Shape (256, 4), EEG segment
        wavelet (str): PyWavelets wavelet name (default: 'db4')
        level (int): Number of DWT decomposition levels (default: 3)
        target_shape (tuple): Shape to reshape each channel's DWT output (freq_bins, time_bins)

    Returns:
        np.ndarray: Shape (4, freq_bins, time_bins)
    """
    assert segment.shape == (256, 4), f"Expected segment shape (256, 4), got {segment.shape}"
    result = []

    for ch in range(4):
        coeffs = pywt.wavedec(segment[:, ch], wavelet=wavelet, level=level)
        # Concatenate all coeff arrays: [A_n, D_n, ..., D1]
        coeff_vec = np.concatenate(coeffs, axis=0)
        # Normalize
        coeff_vec = (coeff_vec - np.mean(coeff_vec)) / (np.std(coeff_vec) + 1e-8)
        # Reshape or pad to target shape
        total_size = target_shape[0] * target_shape[1]
        if coeff_vec.shape[0] < total_size:
            pad = np.zeros(total_size - coeff_vec.shape[0])
            coeff_vec = np.concatenate([coeff_vec, pad])
        coeff_img = coeff_vec[:total_size].reshape(target_shape)
        result.append(coeff_img)

    return np.array(result)  # shape: (4, freq_bins, time_bins)




def extract_sfft_and_dwt(segment, fs=256, sfft_nperseg=64, sfft_noverlap=32, dwt_wavelet='db4', dwt_level=3, target_shape=(33, 5)):
    """
    Extract combined SFFT and DWT features from a single EEG segment.

    Args:
        segment (np.ndarray): Shape (256, 4)
        fs (int): Sampling frequency for SFFT
        sfft_nperseg (int): Segment length for SFFT
        sfft_noverlap (int): Overlap between SFFT windows
        dwt_wavelet (str): Wavelet name for DWT
        dwt_level (int): DWT decomposition level
        target_shape (tuple): Desired output shape (freq_bins, time_bins)

    Returns:
        np.ndarray: Shape (8, freq_bins, time_bins)
    """
    from eeg_features import extract_sfft, extract_dwt

    sfft = extract_sfft(segment, fs=fs, nperseg=sfft_nperseg, noverlap=sfft_noverlap, target_shape=target_shape)
    dwt = extract_dwt(segment, wavelet=dwt_wavelet, level=dwt_level, target_shape=target_shape)

    # Concatenate along channel dimension: (4+4, freq, time)
    combined = np.concatenate([sfft, dwt], axis=0)
    return combined  # shape: (8, freq_bins, time_bins)
