import os
from labels import label_map, validate_labels, format_valid_labels_message
import argparse
import pandas as pd
from datetime import datetime
from scipy.signal import butter, filtfilt
import numpy as np
import normalization
from config import recordEEG, labels_to_orders
import feature_extraction

def check_data(dataset, msg=''):
    if np.isnan(dataset).any():
        nan_flat_index = np.where(np.isnan(dataset.ravel()))[0]
        if len(nan_flat_index) > 0:
            first_nan_flat = nan_flat_index[0]
            # 将扁平索引转换为多维坐标
            first_nan_pos = np.unravel_index(first_nan_flat, dataset.shape)
            np.set_printoptions(suppress=True, precision=100, linewidth=np.inf)
            print(f"first nan: {np.array(dataset[first_nan_pos[0]])}")
        raise ValueError('nan exists in dataset. msg: '+msg)

def parse_labels(label_str, keep_order=False):
    # 使用models.py中的函数验证和转换labels
    selected, abbrs_out, invalid_labels = validate_labels(label_str, keep_order)
    
    # 如果有无效标签，报错并提示
    if invalid_labels:
        error_msg = f"Invalid labels: {', '.join(invalid_labels)}.\n{format_valid_labels_message()}"
        raise ValueError(error_msg)
    return selected, abbrs_out

def segment_data(arr, window_size, sliding_window):
    data_segments=[]
    data_labels=[]

    # calculate taskLen and transitionLen in number of samples
    taskLen=recordEEG.taskLength*recordEEG.hzPerSec;
    transitionLen=recordEEG.transitionLength*recordEEG.hzPerSec;
    maxStartIdx=len(arr)-transitionLen-taskLen+1;
    if(window_size>taskLen):
        raise ValueError(f"window size [{window_size}] is less than taskLength [{taskLen}]!")
    
    cueIdx=0
    cueLen=len(recordEEG.cuesIdx)
    i=0
    while i<maxStartIdx:
        #for each task
        i+=transitionLen # skip the transition stage
        #use the sliding window
        for j in range(i, i+taskLen-window_size+1, sliding_window):
            data_segments.append(arr[j:j+window_size])
            one_hot_label=[0]*cueLen
            one_hot_label[recordEEG.cuesIdx[cueIdx]]=1
            data_labels.append(one_hot_label)
        cueIdx=(cueIdx+1)%cueLen
        i+=taskLen
    return data_segments,data_labels

# @param -doBandPass do we apply the bandpass filter
# @param -normalization do we apply the bandpass filter
def preprocess_files(labels, window_size, sliding_window, lowcut, highcut, doBandPass, featureExtractionMethod, normalizationMethod, user):
    recordEEG.SetStandardCuesAndIdx([lbl for _,lbl in enumerate(labels)],[idx for idx,_ in enumerate(labels)], 0);
    # Find files for each label
    missing_labels = []
    labelname='_'.join(labels);
    total_segments=[];
    total_labels=[];
    
    folder = os.path.join(os.path.dirname(__file__), 'recorded_data', user)
    if os.path.exists(folder):
        files = [f for f in os.listdir(folder) if f.startswith(f"eeg_{labelname}") and f.endswith('.csv')]
        if files:
            for file in files:
                df = pd.read_csv(os.path.join(folder, file))
                # Ignore timestamps
                check_data(df.values, f"processing file {file}. the best way is to delete the file")
                signals = df[["TP9", "AF7", "AF8", "TP10", "Right AUX"]].copy()
                # Subtract Right AUX
                for ch in ["TP9", "AF7", "AF8", "TP10"]:
                    signals[ch] = signals[ch] - signals["Right AUX"]
                signals = signals[["TP9", "AF7", "AF8", "TP10"]]
                if doBandPass==1:
                    # Bandpass filter 5-40Hz
                    fs = 256  # 假设采样率为256Hz，如有不同请修改
                    nyq = 0.5 * fs
                    b, a = butter(4, [lowcut/nyq, highcut/nyq], btype='band')
                    # 对每个通道滤波
                    for ch in signals.columns:
                        signals[ch] = filtfilt(b, a, signals[ch].values)
                arr = np.array(signals.values)
                arr=normalization.normalize(arr, normalizationMethod)
                # Sliding window
                data_segments, data_labels=segment_data(arr, window_size, sliding_window)
                # do fft to data_segments
                if featureExtractionMethod!='none':
                    featExtractResults=[]
                    for segment in data_segments:
                        featExtractResult, freqs=feature_extraction.feature_extract(featureExtractionMethod, segment, lowcut, highcut)
                        featExtractResults.append(featExtractResult);
                    data_segments=featExtractResults
                for segment in data_segments:
                    total_segments.append(segment)
                for lbl in data_labels:
                    total_labels.append(lbl)
    
    return total_segments, total_labels, missing_labels

def main():
    parser = argparse.ArgumentParser(description="EEG Preprocessing Script")
    parser.add_argument("--user", type=str, required=True, help="User identifier for organizing recordings")
    parser.add_argument("-labels", type=str, required=True, help="Comma separated labels to use, e.g. left,right-to-left")
    parser.add_argument("-windowSize", type=int, default=256, help="Window size for segmentation")
    parser.add_argument("-slidingWindow", type=int, default=128, help="Sliding window step")
    parser.add_argument("-bandPass", type=str, default="5,40", help="Bandpass filter range as lowcut,highcut (default: 5,40)")
    parser.add_argument("-asCSV", type=int, default=0, help="save as .csv or .npy")
    parser.add_argument("-doBandPass", type=int, default=1, help="[0 or 1] default as 1. do we apply the bandpass filter")
    parser.add_argument("-normalizationMethod", type=str, default='z-score', help="[none, z-score, min-max, robust] default as z-score. what normalization method do we want?")
    parser.add_argument("-featureExtraction", type=str, default='none', help="[none, fft, sfft, wavelet] default as none. what feature extraction method do we want?")
    args = parser.parse_args()

    labels, abbrs = parse_labels(args.labels, True)
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

    segments, labels_data, missing_labels = preprocess_files(labels, window_size, sliding_window, lowcut, highcut, args.doBandPass, args.featureExtraction, args.normalizationMethod, args.user)
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
    outdir = os.path.join(os.path.dirname(__file__), "preprocessed_data", args.user)
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
                'TP9': [segments[i][j][0] for j in range(len(segments[i]))],
                'TP7': [segments[i][j][1] for j in range(len(segments[i]))],
                'TP8': [segments[i][j][2] for j in range(len(segments[i]))],
                'TP10': [segments[i][j][3] for j in range(len(segments[i]))]
            })
            outname = f"preprocessed_{'_'.join(labels)}[{labels_data[i]}]_{timestr}{i}.csv"
            outpath = os.path.join(outdir, outname)
            df.to_csv(outpath, index=False) # do not save line numbers
            print(f"[INFO] Saved preprocessed data to {outpath}")
    print(f"[INFO] Segments shape: {data_dict['segments'].shape}")
    print(f"[INFO] Labels shape: {data_dict['labels'].shape}")

if __name__ == "__main__":
    main()