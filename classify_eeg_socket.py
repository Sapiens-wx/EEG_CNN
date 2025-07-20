import argparse
import numpy as np
import os
import models
import socket
import json
from labels import validate_labels
from pylsl import StreamInlet, resolve_streams
from scipy.signal import butter, filtfilt

parser = argparse.ArgumentParser(description="EEG Classification Socket Streamer")
parser.add_argument("-model", type=str, required=True, help="Model type: CNN, Transformer, CNN+LSTM, DaViT")
parser.add_argument("-windowSize", type=int, default=256, help="Window size for EEG segments")
parser.add_argument("-slidingWindow", type=int, default=128, help="Sliding window step for EEG segments")
parser.add_argument("-labels", type=str, required=True, help="Comma separated labels for EEG data")
parser.add_argument("-host", type=str, default="127.0.0.1", help="Socket host (default: 127.0.0.1)")
parser.add_argument("-port", type=int, default=9000, help="Socket port (default: 9000)")
parser.add_argument("-bufferRate", type=float, default=2.0, help="Buffer size multiplier relative to windowSize (default: 2.0)")
args = parser.parse_args()

labels_list, _, invalid_labels = validate_labels(args.labels)
if not labels_list:
    print("Error: Labels cannot be empty.")
    exit(1)
if invalid_labels:
    print(f"Error: Invalid labels: {', '.join(invalid_labels)}")
    exit(1)

# Load model
model_dir = os.path.join(os.path.dirname(__file__), 'models')
model_files = [f for f in os.listdir(model_dir) if f.endswith('.keras') and args.model.lower() in f.lower()]
if not model_files:
    print(f"No model file found for type {args.model} in {model_dir}")
    exit(1)
model_path = os.path.join(model_dir, model_files[0])
print(f"Using model: {model_path}")
model = models.LoadModel(
    model_type=args.model,
    windowSize=args.windowSize,
    num_classes=len(labels_list),
    model_optimizer='adam'
)
model.load_weights(model_path)

# Resolve EEG stream
print("Resolving EEG stream...")
streams = resolve_streams()
eeg_streams = [s for s in streams if s.type() == 'EEG']
if not eeg_streams:
    raise RuntimeError("No EEG stream found! Please ensure the Muse device is connected and streaming.")
inlet = StreamInlet(eeg_streams[0])
print("EEG stream resolved. Starting data acquisition...")

# Calculate buffer size
buffer_size = int(args.windowSize * args.bufferRate)
if buffer_size <= args.windowSize:
    print("Error: bufferRate must be greater than 1.0 to ensure effective buffering.")
    exit(1)

window = []
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((args.host, args.port))
    print(f"Connected to {args.host}:{args.port}")
    while True:
        sample, _ = inlet.pull_sample(timeout=1.0)  # 获取一帧EEG数据
        if sample is None:
            continue
        # 预处理：四通道减去Right AUX
        eeg = [sample[0]-sample[4], sample[1]-sample[4], sample[2]-sample[4], sample[3]-sample[4]]
        window.append(eeg)
        if len(window) >= buffer_size:
            # Process only when buffer is full
            segment = np.array(window[:args.windowSize])
            # 滤波 5-40Hz
            fs = 256  # 采样率，如有不同请修改
            lowcut = 5
            highcut = 40
            nyq = 0.5 * fs
            b, a = butter(4, [lowcut/nyq, highcut/nyq], btype='band')
            # 对每个通道滤波
            for ch in range(4):
                segment[:, ch] = filtfilt(b, a, segment[:, ch])
            segment = segment.reshape((1, args.windowSize, 4))
            pred = model.predict(segment)[0]
            msg = json.dumps({"probs": pred.tolist()})
            s.sendall(msg.encode('utf-8') + b'\n')
            print(f"Sent: {msg}")
            window = window[args.slidingWindow:]
