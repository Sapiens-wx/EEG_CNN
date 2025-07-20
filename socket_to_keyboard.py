import argparse
import socket
import json
import time
import keyboard
import os

import subprocess

parser = argparse.ArgumentParser(description="EEG Classification Socket to Keyboard Mapper")
parser.add_argument("-host", type=str, default="127.0.0.1", help="Socket host (default: 127.0.0.1)")
parser.add_argument("-port", type=int, default=9000, help="Socket port (default: 9000)")
parser.add_argument("-threshold", type=float, default=0.7, help="Threshold for classification (default: 0.7)")
parser.add_argument("-modelname", type=str, default=None, help="Model name to get label mapping (default: latest)")
parser.add_argument("-runClassificationToSocket", action="store_true", help="Run classify_eeg_socket.py before connecting")
parser.add_argument("-window", type=int, default=None, help="Window size for classify_eeg_socket.py")
parser.add_argument("-slidingwindow", type=int, default=None, help="Sliding window size for classify_eeg_socket.py")
args = parser.parse_args()

# 可选启动 classify_eeg_socket.py，并传递参数
if args.runClassificationToSocket:
    server_script = os.path.join(os.path.dirname(__file__), "classify_eeg_socket.py")
    if not os.path.exists(server_script):
        print(f"Server script not found: {server_script}")
        exit(1)
    # 检查 window 和 slidingwindow 参数
    if args.window is None or args.slidingwindow is None:
        print("Error: -window 和 -slidingwindow 参数必须提供！")
        exit(1)
    print(f"Starting classify_eeg_socket.py ...")
    cmd = ["python", server_script, f"-window", str(args.window), f"-slidingwindow", str(args.slidingwindow)]
    subprocess.Popen(cmd)

# 获取label名称
label_names = None
model_dir = os.path.join(os.path.dirname(__file__), 'models')
if args.modelname:
    model_files = [f for f in os.listdir(model_dir) if args.modelname in f and f.endswith('.keras')]
    if not model_files:
        print(f"No model file found for name {args.modelname} in {model_dir}")
        exit(1)
    # 从文件名获取label部分
    label_part = model_files[0].split('_')[2].split(',')[0]
    label_names = label_part.split('-')
else:
    # 默认加载最新的
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.keras')]
    if not model_files:
        print(f"No model file found in {model_dir}")
        exit(1)
    model_files.sort(key=lambda x: os.path.getctime(os.path.join(model_dir, x)), reverse=True)
    label_part = model_files[0].split('_')[2].split(',')[0]
    label_names = label_part.split('-')
print(f"Label names: {label_names}")

# 键盘状态管理
key_map = {
    'l': 'left',
    'r': 'right',
    'n': None,
    'l2r': ('left', 'right'),
    'r2l': ('right', 'left'),
    'r2n': ('right', None),
    'l2n': ('left', None)
}
current_state = None

# 连接socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    try:
        s.connect((args.host, args.port))
    except Exception as e:
        print(f"Could not connect to {args.host}:{args.port}: {e}")
        exit(1)
    print(f"Connected to {args.host}:{args.port}")
    buffer = b''
    while True:
        try:
            data = s.recv(4096)
            if not data:
                print("Socket closed.")
                break
            buffer += data
            while b'\n' in buffer:
                line, buffer = buffer.split(b'\n', 1)
                msg = json.loads(line.decode('utf-8'))
                probs = msg['probs']
                # 阈值判断
                valid = [(i, p) for i, p in enumerate(probs) if p >= args.threshold]
                if not valid:
                    # 没有高于阈值，释放所有
                    if current_state:
                        if current_state == 'left':
                            keyboard.release('left')
                        elif current_state == 'right':
                            keyboard.release('right')
                        current_state = None
                    continue
                # 取概率最高的那个
                idx, _ = max(valid, key=lambda x: x[1])
                label = label_names[idx]
                # 状态转换
                if label == 'l':
                    if current_state != 'left':
                        if current_state == 'right':
                            keyboard.release('right')
                        keyboard.press('left')
                        current_state = 'left'
                elif label == 'r':
                    if current_state != 'right':
                        if current_state == 'left':
                            keyboard.release('left')
                        keyboard.press('right')
                        current_state = 'right'
                elif label == 'n':
                    if current_state == 'left':
                        keyboard.release('left')
                    elif current_state == 'right':
                        keyboard.release('right')
                    current_state = None
                elif label in key_map:
                    prev, new = key_map[label]
                    if prev and current_state == prev:
                        keyboard.release(prev)
                    if new:
                        keyboard.press(new)
                    current_state = new
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(0.1)
