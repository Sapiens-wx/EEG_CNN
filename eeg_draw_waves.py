import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    return scipy.signal.filtfilt(b, a, data, axis=0)

def plot_eeg_waves(csv_file_path, output_dir=None):
    """
    画出csv样本里的四种脑波（Delta, Theta, Alpha, Beta）随时间的强度变化。
    假设csv文件的前5列为：时间戳、通道1、通道2、通道3、通道4。
    """
    # 频段定义（Hz）
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30)
    }
    try:
        data = pd.read_csv(csv_file_path)
        if len(data.columns) < 5:
            raise ValueError("CSV文件至少需要5列（时间 + 4通道）")
        time = data.iloc[:, 0]
        time = (time - time.iloc[0]) * 1000  # 转为ms
        eeg = data.iloc[:, 1:5].values  # shape: (n_samples, 4)
        fs = 256  # 假设采样率为256Hz，如有不同请修改
        eeg = bandpass_filter(eeg, 0.5, 40, fs)
        window_size = fs  # 1秒滑窗
        step_size = fs // 4  # 0.25秒步长
        n_windows = (len(eeg) - window_size) // step_size + 1
        band_powers = {band: [] for band in bands}
        times = []
        for i in range(n_windows):
            start = i * step_size
            end = start + window_size
            segment = eeg[start:end, :]
            segment_time = time.iloc[start:end].mean()
            times.append(segment_time)
            # 对每个通道做PSD（加窗）
            psd_all = []
            for ch in range(segment.shape[1]):
                f, Pxx = scipy.signal.welch(segment[:, ch], fs=fs, window='hann', nperseg=window_size, noverlap=0, scaling='density')
                psd_all.append(Pxx)  # shape: (n_freq,)
            psd_all = np.stack(psd_all, axis=1)  # shape: (n_freq, 4)
            for band, (low, high) in bands.items():
                idx = np.where((f >= low) & (f < high))[0]
                if len(idx) == 0:
                    band_powers[band].append(np.nan)
                    continue
                # 计算每个通道该频带的PSD均值
                band_psd = psd_all[idx, :].mean(axis=0)  # (4,)
                # 归一化到带宽
                bandwidth = high - low
                band_psd_norm = band_psd / bandwidth
                # 取平均，转dB
                band_psd_norm_mean = np.mean(band_psd_norm)
                band_power_db = 10 * np.log10(band_psd_norm_mean + 1e-20)  # 避免log(0)
                band_powers[band].append(band_power_db)
        # 绘图
        plt.figure(figsize=(12, 6))
        for band in bands:
            plt.plot(times, band_powers[band], label=band)
        plt.xlabel('Time (ms)')
        plt.ylabel('PSD (dB, µV²/Hz)')
        plt.title(f'EEG Band Power Over Time\n{os.path.basename(csv_file_path)}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # 保存
        if output_dir is None:
            output_dir = os.path.dirname(csv_file_path)
        base = os.path.splitext(os.path.basename(csv_file_path))[0]
        out_path = os.path.join(output_dir, f'{base}_waves.png')
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved: {out_path}")
    except Exception as e:
        print(f"Error processing {csv_file_path}: {str(e)}")

if __name__ == "__main__":
    # 自动处理training_data下所有eeg_*.csv
    data_dir = os.path.join(os.path.dirname(__file__), 'training_data')
    csv_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith('eeg_') and f.endswith('.csv')]
    if not csv_files:
        print("No eeg_*.csv files found in training_data directory.")
    for csv_file in csv_files:
        plot_eeg_waves(csv_file)
