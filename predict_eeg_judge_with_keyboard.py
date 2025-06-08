import numpy as np
import os
from pylsl import StreamInlet, resolve_streams, resolve_bypred
import tensorflow as tf
from pynput.keyboard import Controller, Key, Listener
import time
from preprocess_eeg import preprocess_data

# 全局变量，用于统计
stats = {
    'left': {'correct': 0, 'total': 0},
    'right': {'correct': 0, 'total': 0},
    'rest': {'correct': 0, 'total': 0}
}

# 当前按下的键
current_key = None
should_exit = False

def on_press(key):
    global current_key
    try:
        if key.char == 'q':
            global should_exit
            should_exit = True
            return False
    except AttributeError:
        if key == Key.left:
            current_key = 'left'
        elif key == Key.right:
            current_key = 'right'
        elif key == Key.space:
            current_key = 'rest'

def on_release(key):
    global current_key
    try:
        if key.char == 'q':
            return False
    except AttributeError:
        if (key == Key.left and current_key == 'left') or \
           (key == Key.right and current_key == 'right') or \
           (key == Key.space and current_key == 'rest'):
            current_key = None

def print_stats():
    print("\n=== Performance Statistics ===")
    for action in ['left', 'right', 'rest']:
        correct = stats[action]['correct']
        total = stats[action]['total']
        accuracy = (correct / total * 100) if total > 0 else 0
        print(f"{action.capitalize()}: {correct}/{total} = {accuracy:.2f}% accuracy")

# 加载模型
try:
    model = tf.keras.models.load_model('model.keras')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# 初始化键盘监听器
keyboard = Controller()
listener = Listener(on_press=on_press, on_release=on_release)
listener.start()

# 实时EEG流处理
print("Looking for an EEG stream...")
streams = resolve_bypred("type='EEG'")
inlet = StreamInlet(streams[0])
print("EEG stream found. Starting real-time classification.")

# 实时预测循环
buffer = []
sequence_length = 256
predict_interval = 0.5  # 每0.5秒预测一次

try:
    while not should_exit:
        start_time = time.time()
        
        # 获取新的EEG样本
        samples, _ = inlet.pull_chunk()
        if samples:
            for sample in samples:
                buffer.append(np.array(sample))
                if len(buffer) > sequence_length:
                    buffer.pop(0)

        # 当buffer达到要求长度时进行预测
        if len(buffer) == sequence_length:
            # 预处理数据
            preprocessed_buffer = preprocess_data(buffer)
            sequence = np.array(preprocessed_buffer)
            sequence = np.expand_dims(sequence, axis=0)
            
            # 预测
            outputs = model.predict(sequence, verbose=0)
            prediction = np.argmax(outputs[0])
            
            # 转换预测结果
            predicted_class = 'left' if prediction == 0 else 'right' if prediction == 1 else 'rest'
            
            # 如果有键盘输入，更新统计信息
            if current_key is not None:
                stats[current_key]['total'] += 1
                if current_key == predicted_class:
                    stats[current_key]['correct'] += 1
                    print(f"✓ Correct prediction: {predicted_class} (Ground truth: {current_key})")
                else:
                    print(f"✗ Wrong prediction: {predicted_class} (Ground truth: {current_key})")
            else:
                print(f"Predicted: {predicted_class}")

        # 控制预测间隔
        elapsed_time = time.time() - start_time
        if predict_interval - elapsed_time > 0:
            time.sleep(predict_interval - elapsed_time)

except KeyboardInterrupt:
    print("\nPrediction stopped by user.")
finally:
    listener.stop()
    print_stats()
