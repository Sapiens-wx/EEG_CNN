import numpy as np
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

def feature_extract(featureExtractionMethod, eeg_data, min_frq=0, max_frq=50):
    if featureExtractionMethod=='sfft' :
        raise ValueError("sfft not implemented")
    elif featureExtractionMethod=='wavelet':
        raise ValueError("sfft not implemented")
    elif featureExtractionMethod=='fft':
        return compute_fft(eeg_data, min_frq, max_frq)
    elif featureExtractionMethod!='none':
        raise ValueError(f"feature extraction method [{featureExtractionMethod}] not match")