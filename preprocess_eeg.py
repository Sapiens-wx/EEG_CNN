import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter
import os
import config

# Bandpass filter function
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Applies a bandpass filter to the EEG signal.
    Parameters:
        data: 1D numpy array of EEG signal data (single channel).
        lowcut: Lower cutoff frequency of the filter (Hz).
        highcut: Upper cutoff frequency of the filter (Hz).
        fs: Sampling frequency of the EEG signal (Hz).
        order: The order of the Butterworth filter.
    Returns:
        Filtered EEG signal as a 1D numpy array.
    """
    nyquist = 0.5 * fs  # Nyquist frequency (half the sampling frequency)
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')  # Create filter coefficients
    return lfilter(b, a, data)  # Apply the filter

# Segment function
def segment_data(data):
    """
    Segments EEG data into fixed-size windows with overlapping.
    Parameters:
        data: 2D numpy array (time points x channels).
    Returns:
        segments: 3D numpy array (num_segments x window_size x num_channels).
        segment_labels: 1D numpy array of labels for each segment.
    """
    taskLengthHz=int(config.recordEEG.taskLength*config.recordEEG.hzPerSec);
    transitionLengthHz=int(config.recordEEG.transitionLength*config.recordEEG.hzPerSec);
    windowSizeHz=int(config.recordEEG.windowSize*config.recordEEG.hzPerSec);
    stepSizeHz=int(config.recordEEG.stepSize*config.recordEEG.hzPerSec);
    commandsLength=len(config.recordEEG.commands);

    segments = []
    labels=[]
    for loop in range(0, config.recordEEG.loopCount):
        loopStartHz=taskLengthHz+taskLengthHz*commandsLength*loop;
        for commandIdx in range(0, commandsLength):
            commandStartHz=loopStartHz+taskLengthHz*commandIdx;
            windowStart=commandStartHz+transitionLengthHz-windowSizeHz;
            for i in range(windowStart, commandStartHz+1, stepSizeHz):
                segments.append(data[i:i+windowSizeHz]);
                labels.append(config.CommandEnum2Label(config.recordEEG.commandsEnum[commandIdx]));
    return (np.array(labels), np.array(segments));

# Main preprocessing function. called by predict_eeg.py
def preprocess_data(data, lowcut=1, highcut=50, fs=256):
    """
    Preprocess EEG data from a combined CSV file and save it as NumPy arrays.
    Parameters:
        input_file: Path to the combined CSV file containing raw EEG data and labels.
        save_folder: Folder where the preprocessed data will be saved.
        lowcut: Lower cutoff frequency for the bandpass filter (Hz).
        highcut: Upper cutoff frequency for the bandpass filter (Hz).
        fs: Sampling frequency of the EEG signal (Hz).
    """
    # Apply bandpass filter to each channel
    filtered_data = np.apply_along_axis(bandpass_filter, axis=0, arr=data, lowcut=lowcut, highcut=highcut, fs=fs)
    
    # Normalize each channel (z-score normalization)
    normalized_data = (filtered_data - np.mean(filtered_data, axis=0)) / np.std(filtered_data, axis=0)

    return normalized_data;

# Run the preprocessing script
if __name__ == "__main__":
    # init segments (which will be saved to numpy file)
    total_segments=[];
    total_labels=[];
    # loop through each file in the folder
    data_folder=os.path.join("training_data");
    for file in os.listdir(data_folder):
        if file.endswith(".csv"): # process only CSV files
            df=pd.read_csv(os.path.join(data_folder, file));
            df = df.drop(columns=['timestamps']).values  # Extract EEG channels (all columns except 'Label')
            # preprocess data
            df=preprocess_data(df);
            # Segment data
            labels,segments = segment_data(df)
            total_segments.extend(segments);
            total_labels.extend(labels);
    # Save preprocessed data
    total_segments=np.array(total_segments);
    total_labels=np.array(total_labels);
    print(f"segment shape {total_segments.shape}, label shape {total_labels.shape}");
    for i in range(0, total_segments.shape[0]):
        print(total_segments[i]);
        print(total_labels[i]);
        print("-------------------");
    save_folder = os.path.join("training_data\preprocessed") # CHANGE THIS path to your output folder
    np.save(os.path.join(save_folder, "eeg_segments.npy"), np.array(total_segments))
    np.save(os.path.join(save_folder, "eeg_labels.npy"), np.array(total_labels))
