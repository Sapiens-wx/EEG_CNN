from muselsl import stream, record  # Muse LSL library for streaming and recording EEG data
from datetime import datetime       # For generating timestamps
import os                           # For file and folder handling
import rmdup
import argparse

# Function to start Muse streaming
def start_stream():
    """
    Starts the Muse EEG stream using muselsl.

    This function ensures that the Muse device is actively streaming data.
    If the Muse stream is already running, this function can be skipped.
    """
    try:
        stream()  # Start the Muse stream
    except Exception as e:
        print("Error starting the Muse stream:", e)

# Function to record EEG data
def record_eeg_data(duration, label):
    """
    Records EEG data using muselsl and saves it to a CSV file.

    Parameters:
        duration: int
            Duration of the recording in seconds.
        label: str
            Label to associate with the recording (e.g., "left", "right").
    
    Saves:
        A CSV file named with the label and timestamp in the current working directory.
    """
    # Get the folder where the script is located (current working directory)
    folder = os.path.dirname(os.path.abspath(__file__))
    
    # Generate a unique filename with the label and current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Example: "20231124_153045"
    filename = os.path.join(folder, f"eeg_{label}_{timestamp}.csv")
    
    try:
        # Start recording EEG data
        print(f"Recording EEG data for {duration} seconds...")
        record(duration=duration, filename=filename)  # Save the recording as a CSV
        print(f"Recording saved to {filename}")
        return filename
    except Exception as e:
        print("Error recording EEG data:", e)
        return None

def precise_record(duration, label, enable_progressbar=False):
    """
    更加准确的EEG录制：
    - 先连接EEG流，持续拉取数据到缓冲区
    - 检测到缓冲区有数据后，开始计时
    - 计时期间持续写入数据
    - 时间到后停止写入，保存为CSV

    Parameters:
        duration: int
            Duration of the recording in seconds.
        label: str
            Label to associate with the recording (e.g., "left", "right").
        enable_progressbar: bool
            Whether to display a progress bar during recording.
    
    Saves:
        A CSV file named with the label and timestamp in the current working directory.
    """
    from pylsl import StreamInlet, resolve_streams
    import pandas as pd
    import time
    from datetime import datetime
    import os
    folder = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(folder, f"eeg_{label}_{timestamp}.csv")
    print("[INFO] Using precise buffer method for acquisition...")
    print("Looking for an EEG stream...")
    streams = resolve_streams()
    eeg_streams = [s for s in streams if s.type() == 'EEG']
    if not eeg_streams:
        raise RuntimeError("No EEG stream found!")
    inlet = StreamInlet(eeg_streams[0])
    print("EEG stream found. Waiting for buffer to have data...")
    # Wait for buffer to have data
    while True:
        sample, timestamp = inlet.pull_sample(timeout=0.1)
        if sample is not None:
            break
    print("[INFO] Buffer has data, start timed recording...")
    data = []
    timestamps = []
    start_time = time.time()
    if enable_progressbar:
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

def check_and_start_stream():
    """
    检查是否有EEG流，如果没有则尝试启动一次stream。
    返回True表示有流，False表示无流。
    """
    from pylsl import resolve_streams
    import time
    # First, search for streams
    streams = resolve_streams()
    eeg_streams = [s for s in streams if s.type() == 'EEG']
    if eeg_streams:
        return True
    # No stream, try to start
    try:
        from muselsl import list_muses, stream
        print("[INFO] No EEG stream detected, trying to auto-discover and start muselsl stream ...")
        import threading
        muses = list_muses()
        if not muses:
            print("[ERROR] No Muse device found. Please make sure the device is powered on and near the computer.")
            return False
        muse_address = muses[0]['address']
        print(f"[INFO] Auto-select Muse address: {muse_address}")
        t = threading.Thread(target=stream, args=(muse_address,), daemon=True)
        t.start()
        # Wait a few seconds for the stream to start
        for _ in range(10):
            time.sleep(0.5)
            streams = resolve_streams()
            eeg_streams = [s for s in streams if s.type() == 'EEG']
            if eeg_streams:
                print("[INFO] Successfully detected EEG stream.")
                return True
        print("[ERROR] Still no EEG stream detected after starting stream.")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to start stream: {e}")
        return False

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Muse EEG recording script")
    parser.add_argument("-enableProgress", dest="enable_progressbar", action="store_true", help="Enable recording progress bar")
    parser.add_argument("-t", dest="duration", type=float, default=None, help="Recording duration in seconds (default: ask interactively)")
    parser.add_argument("-lable", dest="lable", type=str, default=None, help="Label for this recording (choose from: left, right, neutral, left to right, right to left, left to neutral, right to neutral, neutral to left, neutral to right)")
    args = parser.parse_args()
    
    allowed_labels = [
        "left", "right", "neutral",
        "left to right", "right to left",
        "left to neutral", "right to neutral",
        "neutral to left", "neutral to right"
    ]

    # Check/start stream
    if not check_and_start_stream():
        print("[FATAL] No EEG stream detected, exiting.")
        exit(1)

    # Step 1: Get label
    label = args.lable
    while label is None or label.strip().lower() not in [l.lower() for l in allowed_labels]:
        print("Available labels:")
        for l in allowed_labels:
            print(f"  - {l}")
        label = input("Enter the label for this recording: ").strip()
        if label.lower() not in [l.lower() for l in allowed_labels]:
            print(f"[ERROR] Invalid label '{label}'. Please choose from the list above.")
            label = None
    # Use the canonical label (case-insensitive match)
    for l in allowed_labels:
        if label.lower() == l.lower():
            label = l
            break

    # Step 2: Get duration
    duration = args.duration
    while duration is None:
        try:
            duration = float(input("Enter the recording duration in seconds: "))
        except Exception:
            print("[ERROR] Please enter a valid number for duration.")
            duration = None

    # Step 3: Record EEG data (use precise_record instead of record_eeg_data)
    filename = precise_record(duration, label, enable_progressbar=args.enable_progressbar)
    
    # Step 4: Remove duplicate lines using Python (instead of .exe)
    if filename and os.path.exists(filename):
        try:
            import pandas as pd
            df = pd.read_csv(filename)
            df_dedup = df.drop_duplicates()
            df_dedup.to_csv(filename, index=False)
            print(f"Removed duplicate lines in {filename}")
        except Exception as e:
            print(f"Error removing duplicates: {e}")
    elif filename:
        print(f"File {filename} does not exist. Skipping duplicate removal.")
