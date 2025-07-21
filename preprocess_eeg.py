import os
from labels import label_map, validate_labels, format_valid_labels_message
import argparse
import pandas as pd
from datetime import datetime
from scipy.signal import butter, filtfilt

def parse_labels(label_str):
    # 使用models.py中的函数验证和转换labels
    selected, abbrs_out, invalid_labels = validate_labels(label_str)
    
    # 如果有无效标签，报错并提示
    if invalid_labels:
        error_msg = f"Invalid labels: {', '.join(invalid_labels)}.\n{format_valid_labels_message()}"
        raise ValueError(error_msg)
    return selected, abbrs_out

# @param -doFilter do we apply the bandpass filter
def preprocess_files(labels, window_size, sliding_window, lowcut, highcut, doFilter):
    # Find files for each label
    data_segments = []
    data_labels = []
    missing_labels = []
    
    for label_idx, lbl in enumerate(labels):
        folder = os.path.join(os.path.dirname(__file__), 'recorded_data')
        if not os.path.exists(folder):
            missing_labels.append(lbl)
            continue
        files = [f for f in os.listdir(folder) if f.startswith(f"eeg_{lbl}") and f.endswith('.csv')]
        if not files:
            missing_labels.append(lbl)
            continue
        for file in files:
            df = pd.read_csv(os.path.join(folder, file))
            # Ignore timestamps
            signals = df[["TP9", "AF7", "AF8", "TP10", "Right AUX"]].copy()
            # Subtract Right AUX
            for ch in ["TP9", "AF7", "AF8", "TP10"]:
                signals[ch] = signals[ch] - signals["Right AUX"]
            signals = signals[["TP9", "AF7", "AF8", "TP10"]]
            if doFilter:
                # Bandpass filter 5-40Hz
                fs = 256  # 假设采样率为256Hz，如有不同请修改
                nyq = 0.5 * fs
                b, a = butter(4, [lowcut/nyq, highcut/nyq], btype='band')
                # 对每个通道滤波
                for ch in signals.columns:
                    signals[ch] = filtfilt(b, a, signals[ch].values)
            # Sliding window
            arr = signals.values
            for start in range(0, len(arr) - window_size + 1, sliding_window):
                window = arr[start:start+window_size]
                data_segments.append(window)
                # Create one-hot label
                one_hot_label = [0] * len(labels)
                one_hot_label[label_idx] = 1
                data_labels.append(one_hot_label)
    
    return data_segments, data_labels, missing_labels

def main():
    parser = argparse.ArgumentParser(description="EEG Preprocessing Script")
    parser.add_argument("-labels", type=str, required=True, help="Comma separated labels to use, e.g. left,right-to-left")
    parser.add_argument("-windowSize", type=int, default=256, help="Window size for segmentation")
    parser.add_argument("-slidingWindow", type=int, default=128, help="Sliding window step")
    parser.add_argument("-bandPass", type=str, default="5,40", help="Bandpass filter range as lowcut,highcut (default: 5,40)")
    parser.add_argument("-asCSV", type=int, default=0, help="save as .csv or .npy")
    parser.add_argument("-doFilter", type=bool, default=True, help="do we apply the bandpass filter")
    args = parser.parse_args()

    labels, abbrs = parse_labels(args.labels)
    window_size = args.windowSize
    sliding_window = args.slidingWindow

    # Parse bandPass argument
    try:
        lowcut, highcut = map(int, args.bandPass.split(","))
        if lowcut <= 0 or highcut <= lowcut:
            raise ValueError
    except ValueError:
        print("[ERROR] Invalid bandPass format. Use two positive integers separated by a comma, e.g., 5,40.")
        return

    # 参数校验
    if window_size <= 0:
        print("[ERROR] windowSize must be > 0.")
        return
    if sliding_window <= 0 or sliding_window > window_size:
        print("[ERROR] slidingWindow must be > 0 and < windowSize.")
        return

    segments, labels_data, missing_labels = preprocess_files(labels, window_size, sliding_window, lowcut, highcut, args.doFilter)
    if missing_labels:
        print("[ERROR] Missing EEG files for labels:")
        for lbl in missing_labels:
            print(f"  - {lbl}")
        print("[ERROR] Aborting: not all requested labels have data.")
        return
    if not segments:
        print("[ERROR] No data segments to save. Exiting.")
        return

    # Save
    outdir = os.path.join(os.path.dirname(__file__), "preprocessed_data")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    abbr_str = '_'.join([label_map[l]["abbr"] for l in labels])
    timestr = datetime.now().strftime('%Y%m%d_%H%M%S')
    import numpy as np
    # Save as dictionary containing both segments and labels
    data_dict = {
        'segments': np.array(segments),
        'labels': np.array(labels_data)
    }
    if args.asCSV==0: # save as npy
        outname = f"preprocessed_{'_'.join(labels)}_{timestr}.npy"
        outpath = os.path.join(outdir, outname)
        np.save(outpath, data_dict, allow_pickle=True)
        print(f"[INFO] Saved preprocessed data to {outpath}")
    else: # save as csv
        # saves at most [args.asCSV] number of csv files
        for i in range(0, min(len(labels_data), args.asCSV)):
            df = pd.DataFrame({
                'TP9': [segments[i][j][0] for j in range(args.windowSize)],
                'TP7': [segments[i][j][1] for j in range(args.windowSize)],
                'TP8': [segments[i][j][2] for j in range(args.windowSize)],
                'TP10': [segments[i][j][3] for j in range(args.windowSize)]
            })
            outname = f"preprocessed_{'_'.join(labels)}[{labels_data[i]}]_{timestr}{i}.csv"
            outpath = os.path.join(outdir, outname)
            df.to_csv(outpath, index=False) # do not save line numbers
            print(f"[INFO] Saved preprocessed data to {outpath}")
    print(f"[INFO] Segments shape: {data_dict['segments'].shape}")
    print(f"[INFO] Labels shape: {data_dict['labels'].shape}")

if __name__ == "__main__":
    main()