import os
import pylsl
import time
import pandas as pd
from datetime import datetime
from labels import validate_labels, format_valid_labels_message

def precise_record(duration, label, show_progressbar):
    #placeholder for the precise_record function
    folder = os.path.join(os.path.dirname(__file__), "recorded_data")
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = os.path.join(folder, f"eeg_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    # Assume EEG device is available, connect directly
    streams = pylsl.resolve_streams()
    eeg_streams = [s for s in streams if s.type() == 'EEG']
    if not eeg_streams:
        raise RuntimeError("No EEG stream found! Please ensure the Muse device is connected and streaming.")
    inlet = pylsl.StreamInlet(eeg_streams[0])
    print("EEG stream found, wait for data...")
    while True:
        sample, timestamp = inlet.pull_sample(timeout=0.1)
        if (sample is not None):
            break
    print("Data received, starting recording...")
    data = []
    timestamps = []
    start_time = time.time()
    if show_progressbar:
        try:
            from tqdm import tqdm
            bar = tqdm(
                total=duration,
                desc="Recording...",
                ncols=53,  # 1/3 shorter than original
                dynamic_ncols=True,
                mininterval=0.1,
                bar_format='{l_bar}{bar}| {desc} {postfix}'
            )
        except ImportError:
            print("[WARN] tqdm not installed, progress bar will not be shown. Only 'Recording...' will be displayed.")
            bar = None
    else:
        print("Recording...", flush=True)
        bar = None
    last_update = start_time
    while (time.time() - start_time) < duration:
        sample, timestamp = inlet.pull_sample(timeout=1.0)
        if sample is not None:
            data.append(sample)
            timestamps.append(timestamp)
        now = time.time()
        if bar and (now - last_update >= 0.05 or now - start_time >= duration):
            elapsed = now - start_time
            bar.n = min(duration, elapsed)
            # Show x.xx/total on the right
            bar.set_description(f"Remaining(s): {max(0, duration-elapsed):.2f}")
            bar.set_postfix({"Progress": f"{elapsed:.2f}/{duration:.2f}"}, refresh=False)
            bar.refresh()
            last_update = now
    if bar:
        bar.n = duration
        bar.set_description("Remaining(s): 0.00")
        bar.set_postfix({"Progress": f"{duration:.2f}/{duration:.2f}"}, refresh=False)
        bar.refresh()
        bar.close()
    print("[INFO] Recording finished, writing to file...")
    df = pd.DataFrame(data, columns=["TP9", "AF7", "AF8", "TP10", "Right AUX"])
    df.insert(0, "timestamps", timestamps)
    df.to_csv(filename, index=False)
    print(f"[INFO] Saved to {filename}")
    # Analysis
    if len(df) > 1:
        t_start = df['timestamps'].iloc[0]
        t_end = df['timestamps'].iloc[-1]
        print(f"[INFO] Data timestamp range: {t_start} ~ {t_end} (span: {t_end-t_start:.3f} s)")
        print(f"[INFO] Average sampling rate: {len(df)/(t_end-t_start):.1f} Hz")
    else:
        print("[WARN] Too few data rows, cannot analyze time span")
    return filename

def check_blue_muse_stream():
    from pylsl import resolve_streams
    import time
    streams = resolve_streams()
    eeg_streams = [s for s in streams if s.type() == 'EEG']
    if eeg_streams:
        return True
    else:
        print("[WARN] No EEG stream found, please ensure the Muse device is connected and streaming.")
        return False


# Main function to record EEG data
if __name__ == "__main__":
    import argparse
    import re
    parser = argparse.ArgumentParser(description="Muse EEG recording script")
    parser.add_argument("-duration", type = int, default = 10, help = "Duration of recording in seconds")
    parser.add_argument("-label", type = str, default = None, help = "Label for the recording")
    parser.add_argument("-progressbar", action = "store_true", help = "Show progress bar during recording")
    args = parser.parse_args()

    valid_labels = [
        "left", "right", "neutral",
        "left-to-right", "right-to-left",
        "left-to-neutral", "right-to-neutral",
        "neutral-to-left", "neutral-to-right"
    ]

    if not check_blue_muse_stream():
        print("[ERROR] No Muse EEG stream found. Exiting...")
        exit(1)

    # Step 1: Validate and normalize label
    lable = args.label
    while True:
        if lable is None:
            lable = input("Enter label for recording: ")
        labels_list, _, invalid_labels = validate_labels(lable)
        if not labels_list:
            print("[WARN] Label cannot be empty.")
            lable = None
            continue
        if invalid_labels:
            print(f"[WARN] Invalid label(s): {', '.join(invalid_labels)}")
            print(format_valid_labels_message())
            lable = None
            continue
        # 只取第一个有效label
        lable = labels_list[0]
        break

    # Step 2: Get recording duration
    duration = args.duration
    while True:
        if isinstance(duration, int) and duration > 0:
            break
        user_input = input("Enter recording duration in seconds (default 10): ")
        if not user_input.strip():
            duration = 10
            break
        if user_input.isdigit() and int(user_input) > 0:
            duration = int(user_input)
            break
        print("[ERROR] Invalid duration. Please enter a positive integer.")
    
    # Step 3: Start recording
    filename = precise_record(duration, lable, args.progressbar)

    if filename and os.path.exists(filename):
        try:
            import pandas as pd
            # Remove duplicate lines using pandas
            df = pd.read_csv(filename)
            df.drop_duplicates(inplace=True)
            df.to_csv(filename, index=False)
            print(f"[INFO] Duplicates removed, updated file saved to {filename}")
        except ImportError:
            print("[WARN] pandas not installed, cannot remove duplicates. Please install pandas to enable this feature.")
    elif filename:
        print("[ERROR] Recording failed or file not found.")