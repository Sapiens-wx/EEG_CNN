import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch

def get_files(file_path=None, recorded_data_dir="recorded_data"):
    if file_path:
        # If a specific file path is provided, return it in a list if it's a CSV file
        if os.path.isfile(file_path) and file_path.lower().endswith('.csv'):
            return [file_path]
        else:
            raise FileNotFoundError(f"The CSV file {file_path} does not exist.")
    else:
        # Otherwise, list all CSV files in the recorded_data directory (non-recursive)
        if os.path.exists(recorded_data_dir):
            return [
                os.path.join(recorded_data_dir, f)
                for f in os.listdir(recorded_data_dir)
                if os.path.isfile(os.path.join(recorded_data_dir, f)) and f.lower().endswith('.csv')
            ]
        else:
            raise FileNotFoundError(f"The directory {recorded_data_dir} does not exist.")

def filter_signals(file_path, lowcut=5, highcut=40, fs=256):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        with open(file_path, 'r') as f:
            print("File content preview:")
            for i, line in enumerate(f):
                print(line.strip())
                if i >= 9:  # Print first 10 lines
                    break
        raise

    signals = df[["TP9", "AF7", "AF8", "TP10", "Right AUX"]].copy()
    for ch in ["TP9", "AF7", "AF8", "TP10"]:
        signals[ch] = signals[ch] - signals["Right AUX"]
    signals = signals[["TP9", "AF7", "AF8", "TP10"]]
    nyq = 0.5 * fs
    b, a = butter(4, [lowcut / nyq, highcut / nyq], btype='band')
    for ch in signals.columns:
        signals[ch] = filtfilt(b, a, signals[ch].values)
    return signals

def compute_psd(signals, fs=256):
    """
    Compute Power Spectral Density for the entire signal
    """
    psd_data = {}
    signals_array = signals.to_numpy()
    freqs, psd = welch(signals_array, fs=fs, axis=0)
    psd_data['freqs'] = freqs
    psd_data['psd'] = psd
    return psd_data

def plot_brainwave_bands(signals, file_name, window_size=256, overlap=128, fs=256, time_scale=1.0):
    """
    Plot real-time brainwave band power over time
    time_scale: Scale factor for horizontal axis (larger values stretch the time axis)
    """
    freq_bands = {
        'Theta (4-8 Hz)': (4, 8),
        'Alpha (8-13 Hz)': (8, 13),
        'Beta (13-30 Hz)': (13, 30),
        'Gamma (30-40 Hz)': (30, 40)
    }
    
    signals_array = signals.to_numpy()
    
    # Calculate number of windows
    num_windows = (signals_array.shape[0] - window_size) // overlap + 1
    band_powers = {band: [] for band in freq_bands}
    time_points = []
    
    for i in range(num_windows):
        start_idx = i * overlap
        end_idx = start_idx + window_size
        
        if end_idx > signals_array.shape[0]:
            break
            
        segment = signals_array[start_idx:end_idx, :]
        
        # Calculate PSD for this segment
        freqs, psd = welch(segment, fs=fs, axis=0)
        
        # Calculate band powers (average across all channels)
        for band, (low, high) in freq_bands.items():
            band_mask = (freqs >= low) & (freqs <= high)
            band_power = np.mean(psd[band_mask, :])  # Average across frequency and channels
            band_powers[band].append(band_power)
        
        # Time point in seconds (adjusted by time_scale)
        time_points.append((start_idx / fs) / time_scale)
    
    # Calculate figure width based on time_scale for better readability
    figure_width = max(12, 12 * time_scale)
    
    # Plot the results
    plt.figure(figsize=(figure_width, 8))
    for band, powers in band_powers.items():
        plt.plot(time_points, powers, label=band, linewidth=2)
    
    plt.title(f"Real-Time Brainwave Band Power - {file_name}")
    plt.xlabel(f"Time (s, scale: {time_scale:.1f}x)")
    plt.ylabel("Power (µV²/Hz)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_dir = "brainwave_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f"{file_name}_brainwaves.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved brainwave plot to {output_path}")
    plt.close()

def get_global_min_max(files, lowcut, highcut):
    """
    Calculate global min/max for consistent scaling (not used in new visualization)
    """
    global_min, global_max = float('inf'), float('-inf')
    for file in files:
        signals = filter_signals(file, lowcut, highcut)
        global_min = min(global_min, signals.min().min())
        global_max = max(global_max, signals.max().max())
    return global_min, global_max

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize EEG brainwave bands over time.")
    parser.add_argument("-filePath", type=str, help="Relative or absolute path to a specific file.", default=None)
    parser.add_argument("-lowcut", type=int, default=4, help="Low cut frequency for bandpass filter (default: 4 Hz)")
    parser.add_argument("-highcut", type=int, default=40, help="High cut frequency for bandpass filter (default: 40 Hz)")
    parser.add_argument("-windowSize", type=int, default=256, help="Window size for analysis (default: 256)")
    parser.add_argument("-overlap", type=int, default=128, help="Overlap between windows (default: 128)")
    parser.add_argument("-timeScale", type=float, default=1.0, help="Scale factor for horizontal axis to improve readability (default: 1.0)")
    args = parser.parse_args()

    try:
        files = get_files(args.filePath)
        print(f"Processing {len(files)} file(s)...")
        
        for file in files:
            print(f"Processing: {file}")
            signals = filter_signals(file, args.lowcut, args.highcut)
            plot_brainwave_bands(signals, os.path.basename(file), args.windowSize, args.overlap, time_scale=args.timeScale)
            
    except Exception as e:
        print("Error:", e)

