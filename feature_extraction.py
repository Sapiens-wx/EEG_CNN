import numpy as np
import scipy
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

def compute_fft(eeg_data, min_frq=0, max_freq=50, sampling_rate=256):
    """
    对 EEG 数据进行 FFT 变换，仅保留 0-max_freq Hz 的频率成分
    
    Args:
        eeg_data: 形状为 (samples, channels) 的 EEG 数据
        sampling_rate: 采样率 (Hz)
        max_freq: 保留的最大频率 (Hz)
        
    Returns:
        (fft_result, freqs):
            fft_result: 形状为 (n_freqs, channels) 的复数 FFT 结果
            freqs: 频率轴数组 (0 ~ max_freq)
    """
    n_samples = eeg_data.shape[0]
    
    # 计算 FFT (沿时间轴)
    fft_result = fft(eeg_data, axis=0)
    
    # 计算频率轴
    freqs = fftfreq(n_samples, d=1/sampling_rate)
    
    # 仅保留 min_frq ~ max_freq 的频率成分
    mask = (freqs >= min_frq) & (freqs <= max_freq)
    freqs = freqs[mask]
    fft_result = fft_result[mask, :]
    
    return np.abs(fft_result), freqs

def extract_waves(segment, fs=256, window_size=64, nperseg=None, noverlap=None):
    """
    计算 EEG 信号的频带功率（Alpha, Beta, Theta, Delta）
    
    参数:
        segment : numpy.ndarray, shape=(n_samples, n_channels)
            输入 EEG 数据
        fs : int, 默认 256
            采样频率 (Hz)
        window_size : int, 默认 64
            滑动窗口大小（样本数）
        nperseg : int, 可选
            STFT 的窗口长度（默认等于 window_size）
        noverlap : int, 可选
            STFT 的重叠样本数（默认 window_size // 2）
    
    返回:
        band_powers : numpy.ndarray, shape=(x, 4)
            每行是一个时间窗口，每列是 [Delta, Theta, Alpha, Beta] 的功率 (dB)
    """
    n_samples, n_channels = segment.shape
    if nperseg is None:
        nperseg = window_size
    if noverlap is None:
        noverlap = nperseg // 2  # 默认 50% 重叠

    # 定义频带 (单位: Hz)
    bands = [
        #('Delta', 2, 4),
        ('Theta', 4, 8),
        #('Alpha', 8, 13),
        ('Beta', 13, 30)
    ]
    n_bands = len(bands)

    # 计算总窗口数
    step = nperseg - noverlap
    n_windows = (n_samples - noverlap) // step
    band_powers = np.zeros((n_windows, n_bands))

    for win_idx in range(n_windows):
        # 获取当前窗口数据
        start = win_idx * step
        end = start + nperseg
        if end > n_samples:
            break
        window_data = segment[start:end, :]  # shape: (nperseg, n_channels)

        # 初始化当前窗口的频带功率 (n_channels, n_bands)
        win_band_powers = np.zeros((n_channels, n_bands))

        for ch in range(n_channels):
            # 计算 STFT
            f, t, Zxx = scipy.signal.stft(
                window_data[:, ch],
                fs=fs,
                window='hann',
                nperseg=nperseg,
                noverlap=noverlap,
                boundary=None
            )
            psd = np.abs(Zxx) ** 2  # 功率谱密度

            # 计算各频带功率
            for b, (_, low, high) in enumerate(bands):
                band_mask = (f >= low) & (f < high)
                if not np.any(band_mask):
                    win_band_powers[ch, b] = np.nan
                else:
                    band_psd = psd[band_mask, :].mean(axis=0)  # 频带内平均功率
                    bandwidth = high - low
                    win_band_powers[ch, b] = 10 * np.log10(band_psd.mean() / bandwidth + 1e-12)

        # 取所有通道的平均
        band_powers[win_idx, :] = np.nanmean(win_band_powers, axis=0)

    return band_powers, None

def feature_extract(featureExtractionMethod, eeg_data, min_frq=0, max_frq=50):
    if featureExtractionMethod=='sfft' :
        raise ValueError("sfft not implemented")
    elif featureExtractionMethod=='wavelet':
        raise ValueError("wavelet not implemented")
    elif featureExtractionMethod=='fft':
        return compute_fft(eeg_data, min_frq, max_frq)
    elif featureExtractionMethod=='wave':
        return extract_waves(eeg_data)
    elif featureExtractionMethod!='none':
        raise ValueError(f"feature extraction method [{featureExtractionMethod}] not match")