import numpy as np
import os
from pylsl import StreamInlet, resolve_streams, resolve_bypred
import tensorflow as tf
from pynput.keyboard import Controller, Key, Listener
import time
from preprocess_eeg import preprocess_data
import sys

# 全局变量，用于统计
stats = {
    'left': {'correct': 0, 'total': 0},
    'right': {'correct': 0, 'total': 0},
    'rest': {'correct': 0, 'total': 0}
}

# 当前按下的键
current_key = None
should_exit = False

# 9-category definitions
CATEGORY_NAMES = [
    "left", "right", "neutral",
    "left to right", "right to left",
    "left to neutral", "right to neutral",
    "neutral to left", "neutral to right"
]
CATEGORY_IDX_TO_NAME = {idx: name for idx, name in enumerate(CATEGORY_NAMES)}
CATEGORY_NAME_TO_IDX = {name: idx for idx, name in enumerate(CATEGORY_NAMES)}

# 用sys.argv解析--transitionTest参数
transition_test_mode = '--transitionTest' in sys.argv

# Instruction for transitionTest mode
TRANSITION_INSTRUCTION = '''\
=== Transition Test Mode Instructions ===
1. 按下一个方向键（左/右/空格）后，当前状态设为该方向（左/右/中性）。
2. 若要测试转变（如左到右），请长按另一个方向键（如右）。
   - 程序会开始检测EEG输出是否出现了对应的transition label（如left to right）。
   - 检测到后会提示“转变成功”，并将当前状态切换为目标方向。
   - 若松开按键前未检测到transition，则提示“转变失败”，状态不变。
3. 只有transition label出现时才允许状态变更。
4. 只能通过严格的transition（如左->left to right->右），不能直接左变右。
5. 松开按键没有检测到transition视为失败。
6. 按q退出。
========================================
'''

if transition_test_mode:
    print(TRANSITION_INSTRUCTION)
    # Transition test mode
    global current_state, transition_target, transition_detecting, transition_success, pressed_key
    current_state = 'neutral'  # 初始状态
    transition_target = None
    transition_detecting = False
    transition_success = False
    pressed_key = None
    should_exit = False
    print(f"当前状态: {current_state}")

    def on_press(key):
        global current_state, transition_target, transition_detecting, pressed_key, transition_success, should_exit
        try:
            if key.char == 'q':
                should_exit = True
                return False
        except AttributeError:
            if not transition_detecting:
                # 第一次按下，设定当前状态
                if key == Key.left:
                    current_state = 'left'
                    print("[INFO] 按下左键，当前状态设为left")
                elif key == Key.right:
                    current_state = 'right'
                    print("[INFO] 按下右键，当前状态设为right")
                elif key == Key.space:
                    current_state = 'neutral'
                    print("[INFO] 按下空格，当前状态设为neutral")
                pressed_key = key
            # 第二次长按不同方向键，进入transition检测
            if pressed_key is not None and not transition_detecting:
                if (key == Key.left and current_state != 'left') or (key == Key.right and current_state != 'right') or (key == Key.space and current_state != 'neutral'):
                    transition_detecting = True
                    transition_success = False
                    if key == Key.left:
                        transition_target = 'left'
                    elif key == Key.right:
                        transition_target = 'right'
                    elif key == Key.space:
                        transition_target = 'neutral'
                    print(f"[INFO] 开始检测transition: {current_state} -> {transition_target}，请保持按键直到检测到transition label！")

    def on_release(key):
        global transition_detecting, transition_success, current_state, transition_target, pressed_key
        try:
            if key.char == 'q':
                return False
        except AttributeError:
            if transition_detecting:
                if transition_success:
                    print(f"[SUCCESS] 检测到transition，状态已切换为{transition_target}")
                    current_state = transition_target
                else:
                    print(f"[FAIL] 未检测到transition，状态保持为{current_state}")
                transition_detecting = False
                transition_target = None
                pressed_key = None
            else:
                pressed_key = None

    # 重新初始化键盘监听器
    listener = Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # 加载模型
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model('model.keras')
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

    # 实时EEG流处理
    print("Looking for an EEG stream...")
    streams = resolve_bypred("type='EEG'")
    inlet = StreamInlet(streams[0])
    print("EEG stream found. Starting transition test mode.")

    buffer = []
    sequence_length = 256
    predict_interval = 0.5

    try:
        while not should_exit:
            start_time = time.time()
            samples, _ = inlet.pull_chunk()
            if samples:
                for sample in samples:
                    buffer.append(np.array(sample))
                    if len(buffer) > sequence_length:
                        buffer.pop(0)
            if len(buffer) == sequence_length and transition_detecting and transition_target is not None:
                preprocessed_buffer = preprocess_data(buffer)
                sequence = np.array(preprocessed_buffer)
                sequence = np.expand_dims(sequence, axis=0)
                outputs = model.predict(sequence, verbose=0)
                prediction = np.argmax(outputs[0])
                predict_label = CATEGORY_IDX_TO_NAME.get(prediction, str(prediction))
                print(f"[EEG] Detected label: {predict_label}")
                # 判断transition
                transition_map = {
                    ('left', 'right'): 'left to right',
                    ('right', 'left'): 'right to left',
                    ('left', 'neutral'): 'left to neutral',
                    ('right', 'neutral'): 'right to neutral',
                    ('neutral', 'left'): 'neutral to left',
                    ('neutral', 'right'): 'neutral to right',
                }
                expected_transition = transition_map.get((current_state, transition_target))
                if predict_label == expected_transition:
                    transition_success = True
            elapsed_time = time.time() - start_time
            if predict_interval - elapsed_time > 0:
                time.sleep(predict_interval - elapsed_time)
    except KeyboardInterrupt:
        print("\nTransition test stopped by user.")
    finally:
        listener.stop()
        print("[INFO] Transition test mode exited.")
    exit(0)

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
