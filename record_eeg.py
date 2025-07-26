import os
import pylsl
import time
import pandas as pd
from datetime import datetime
from labels import validate_labels, format_valid_labels_message
from config import recordEEG
from tqdm import tqdm

def precise_record(duration, label_name, show_progressbar):
    #placeholder for the precise_record function
    folder = os.path.join(os.path.dirname(__file__), "recorded_data")
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = os.path.join(folder, f"eeg_{label_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
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
    duration = recordEEG.duration
    cue_duration = recordEEG.taskLength + recordEEG.transitionLength
    total_cue_phases = len(recordEEG.cues) * recordEEG.loopCount
    
    # Initialize progress bars
    if show_progressbar:
        try:
            # Main recording progress bar (duration)
            duration_bar = tqdm(
                total=duration,
                desc="Recording...",
                ncols=80,
                dynamic_ncols=True,
                mininterval=0.1,
                bar_format='{l_bar}{bar}| {desc} {postfix}',
                position=0
            )
            
            # Cue progress bar will be handled manually
            cue_bar = None
        except ImportError:
            print("[WARN] tqdm not installed, progress bar will not be shown.")
            duration_bar = None
    else:
        print("Recording...", flush=True)
        duration_bar = None
    
    last_update = start_time
    current_cue_index = 0
    current_loop = 0
    
    while (time.time() - start_time) < duration:
        # Get current time and elapsed time
        now = time.time()
        elapsed = now - start_time
        
        # Determine current cue phase
        cue_phase = int(elapsed // cue_duration)
        current_cue_index = cue_phase % len(recordEEG.cues)
        current_loop = cue_phase // len(recordEEG.cues)
        
        # Get current cue
        current_cue = recordEEG.cues[current_cue_index]
        
        # Build cue progress bar (manual implementation)
        cue_bar_str = "[{}] |".format(current_cue)
        cue_progress_width=30;
        num_cue = len(recordEEG.cues)
        segment_width = int(cue_progress_width//num_cue)  # Total width for cue progress bar
        
        for i in range(num_cue):
            if i == current_cue_index:
                # Current active segment
                cue_bar_str += "█" * segment_width
            else:
                # Inactive segments
                cue_bar_str += " " * segment_width
        
        cue_bar_str += "|"
        
        # Pull sample
        sample, timestamp = inlet.pull_sample(timeout=1.0)
        if sample is not None:
            data.append(sample)
            timestamps.append(timestamp)
        
        # Update progress bars
        if duration_bar and (now - last_update >= 0.05 or now - start_time >= duration):
            # Update duration progress
            duration_bar.n = min(duration, elapsed)
            duration_bar.set_postfix({
                "Loop": f"{current_loop+1}/{recordEEG.loopCount}",
                "Cue": cue_bar_str
            }, refresh=False)
            duration_bar.refresh()
            last_update = now
    
    # Final update
    if duration_bar:
        duration_bar.n = duration
        duration_bar.set_description("Recording complete")
        duration_bar.set_postfix({
            "Progress": f"{duration:.2f}/{duration:.2f}",
            "Loop": f"{recordEEG.loopCount}/{recordEEG.loopCount}"
        }, refresh=False)
        duration_bar.refresh()
        duration_bar.close()
    
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
    parser = argparse.ArgumentParser(description="Muse EEG recording script")
    parser.add_argument("-cues", type = str, default = "l,r,n", help = "an array of cues you want to generate")
    parser.add_argument("-loop", type = int, default = 4, help = "how many times do you want to repeat the array of cues")
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
    cues=args.cues.split(',')

    # Step 2: setup cues and loop to recordEEG
    if args.loop<=0:
        raise ValueError("loop cannot <= 0")
    recordEEG.SetCues(cues, args.loop);
    # 多标签全部写进文件名，使用标准名并用_分隔
    labelname = '_'.join(recordEEG.cues)
    print(f"[INFO] cues={recordEEG.cues}, duration={recordEEG.duration}")
    duration = recordEEG.duration;
    if duration <= 0:
        raise ValueError("[ERROR] Invalid duration. Please enter a positive integer in config.py")
    
    # Step 3: Start recording
    filename = precise_record(duration, labelname, True)

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