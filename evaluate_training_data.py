import numpy as np
import os
import pandas as pd
from preprocess_eeg import preprocess_data
import tensorflow as tf
import argparse
import concurrent.futures
from tqdm import tqdm
import multiprocessing

# Load the model
try:
    model = tf.keras.models.load_model('model.keras')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

def predict_file(file_path, n_predictions=10):
    # Load and preprocess the file
    df = pd.read_csv(file_path)
    data = df.drop(columns=['timestamps']).values
    
    # Process in windows of 256 samples
    sequence_length = 256
    all_window_predictions = []
    
    for i in range(0, len(data) - sequence_length + 1, sequence_length):
        window = data[i:i + sequence_length]
        if len(window) == sequence_length:
            window_predictions = []
            # Preprocess the window
            preprocessed_window = preprocess_data(window)
            sequence = np.array(preprocessed_window)
            
            # Make n predictions for this window
            sequence = np.expand_dims(sequence, axis=0)
            for _ in range(n_predictions):
                outputs = model.predict(sequence, verbose=0)
                prediction = np.argmax(outputs[0])
                window_predictions.append(prediction)
            
            # Get most common prediction for this window
            most_common = max(set(window_predictions), key=window_predictions.count)
            all_window_predictions.append(most_common)
            
            # Print detailed prediction results for this window
            left_count = window_predictions.count(0)
            right_count = window_predictions.count(1)
            rest_count = window_predictions.count(2)
            print(f"  Window {i//sequence_length + 1}: Left={left_count}, Right={right_count}, Rest={rest_count} -> {most_common}")
    
    # Return most common prediction across all windows
    if all_window_predictions:
        final_prediction = max(set(all_window_predictions), key=all_window_predictions.count)
        
        # Calculate overall statistics
        total_left = sum(1 for p in all_window_predictions if p == 0)
        total_right = sum(1 for p in all_window_predictions if p == 1)
        total_rest = sum(1 for p in all_window_predictions if p == 2)
        total_windows = len(all_window_predictions)
        
        print(f"  Overall: Left={total_left}/{total_windows}, Right={total_right}/{total_windows}, Rest={total_rest}/{total_windows}")
        return final_prediction
    return -1

def predict_file_with_path(file_path):
    import tensorflow as tf
    from preprocess_eeg import preprocess_data
    import pandas as pd
    import numpy as np
    df = pd.read_csv(file_path)
    data = df.drop(columns=['timestamps']).values
    sequence_length = 256
    all_window_predictions = []
    model = tf.keras.models.load_model('model.keras')
    for i in range(0, len(data) - sequence_length + 1, sequence_length):
        window = data[i:i + sequence_length]
        if len(window) == sequence_length:
            preprocessed_window = preprocess_data(window)
            sequence = np.array(preprocessed_window)
            sequence = np.expand_dims(sequence, axis=0)
            outputs = model.predict(sequence, verbose=0)
            prediction = np.argmax(outputs[0])
            all_window_predictions.append(prediction)
    if all_window_predictions:
        final_prediction = max(set(all_window_predictions), key=all_window_predictions.count)
        class_name = "left" if final_prediction == 0 else "right" if final_prediction == 1 else "rest"
        return (os.path.basename(file_path), class_name, final_prediction)
    return (os.path.basename(file_path), "unknown", -1)

def predict_file_with_path_count(file_path):
    import tensorflow as tf
    from preprocess_eeg import preprocess_data
    import pandas as pd
    import numpy as np
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-testCount", type=int, default=10)
    args, _ = parser.parse_known_args()
    df = pd.read_csv(file_path)
    data = df.drop(columns=['timestamps']).values
    sequence_length = 256
    n_predictions = args.testCount
    all_window_predictions = []
    model = tf.keras.models.load_model('model.keras')
    for i in range(0, len(data) - sequence_length + 1, sequence_length):
        window = data[i:i + sequence_length]
        if len(window) == sequence_length:
            window_predictions = []
            preprocessed_window = preprocess_data(window)
            sequence = np.array(preprocessed_window)
            sequence = np.expand_dims(sequence, axis=0)
            for _ in range(n_predictions):
                outputs = model.predict(sequence, verbose=0)
                prediction = np.argmax(outputs[0])
                window_predictions.append(prediction)
            all_window_predictions.extend(window_predictions)
    # 统计
    left_count = all_window_predictions.count(0)
    right_count = all_window_predictions.count(1)
    rest_count = all_window_predictions.count(2)
    total = len(all_window_predictions)
    # 判断真实类别
    file_base = os.path.basename(file_path)
    if "left" in file_base.lower():
        true_class = 0
    elif "right" in file_base.lower():
        true_class = 1
    elif "rest" in file_base.lower():
        true_class = 2
    else:
        true_class = -1
    correct = all_window_predictions.count(true_class) if true_class != -1 else 0
    # 返回详细统计
    return (file_base, true_class, left_count, right_count, rest_count, total, correct)

def predict_file_with_path_count_worker(file_path, window_size, test_count, idx, queue):
    import tensorflow as tf
    from preprocess_eeg import preprocess_data
    import pandas as pd
    import numpy as np
    df = pd.read_csv(file_path)
    data = df.drop(columns=['timestamps']).values
    n_windows = (len(data) - window_size) // window_size + 1 if len(data) >= window_size else 0
    model = tf.keras.models.load_model('model.keras')
    file_base = os.path.basename(file_path)
    if "left" in file_base.lower():
        true_class = 0
    elif "right" in file_base.lower():
        true_class = 1
    elif "rest" in file_base.lower():
        true_class = 2
    else:
        true_class = -1
    acc_list = []
    left_total = 0
    right_total = 0
    rest_total = 0
    for t in range(test_count):
        window_predictions = []
        for i in range(0, len(data) - window_size + 1, window_size):
            window = data[i:i + window_size]
            if len(window) == window_size:
                preprocessed_window = preprocess_data(window)
                sequence = np.array(preprocessed_window)
                sequence = np.expand_dims(sequence, axis=0)
                outputs = model.predict(sequence, verbose=0)
                prediction = np.argmax(outputs[0])
                window_predictions.append(prediction)
            queue.put((idx, 1))
        correct = window_predictions.count(true_class) if true_class != -1 else 0
        acc = correct / len(window_predictions) if window_predictions else 0
        acc_list.append(acc)
        left_total += window_predictions.count(0)
        right_total += window_predictions.count(1)
        rest_total += window_predictions.count(2)
    acc_mean = np.mean(acc_list)
    acc_std = np.std(acc_list)
    total = left_total + right_total + rest_total
    queue.put((idx, (file_base, true_class, left_total, right_total, rest_total, total, acc_mean, acc_std)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--jobs", type=int, default=1, help="并发进程数")
    parser.add_argument("-testCount", type=int, default=5, help="每个样本的测试次数")
    parser.add_argument("-windowTime", type=int, default=500, help="窗口时间长度（毫秒）")
    args = parser.parse_args()

    training_data_dir = "training_data"
    files = [os.path.join(training_data_dir, f) for f in os.listdir(training_data_dir) if f.endswith(".csv")]
    results = []
    all_total = 0
    all_left = 0
    all_right = 0
    all_rest = 0
    all_accs = []
    window_size = int(args.windowTime * 256 / 1000)

    if args.jobs > 1:
        manager = multiprocessing.Manager()
        queue = manager.Queue()
        n_windows_list = []
        for file in files:
            df = pd.read_csv(file)
            data = df.drop(columns=['timestamps']).values
            n_windows = (len(data) - window_size) // window_size + 1 if len(data) >= window_size else 0
            n_windows_list.append(n_windows * args.testCount)
        bars = [tqdm(total=n_windows_list[i], desc=os.path.basename(files[i]), position=i) for i in range(len(files))]
        pool = multiprocessing.Pool(processes=args.jobs)
        for idx, file in enumerate(files):
            pool.apply_async(predict_file_with_path_count_worker, (file, window_size, args.testCount, idx, queue))
        finished = 0
        file_results = [None] * len(files)
        while finished < len(files):
            idx, data = queue.get()
            if isinstance(data, tuple):
                file_results[idx] = data
                finished += 1
                bars[idx].close()
            else:
                bars[idx].update(data)
        pool.close()
        pool.join()
        results = file_results
    else:
        for file in files:
            print(f"\nProcessing {os.path.basename(file)}:")
            result = predict_file_with_path_count(file, window_size, args.testCount)
            results.append(result)

    print("\nSummary of predictions:")
    left_accs = []
    right_accs = []
    rest_accs = []
    for file, true_class, left_count, right_count, rest_count, total, acc_mean, acc_std in results:
        if true_class == 0:
            true_str = "left"
            left_accs.append(acc_mean)
        elif true_class == 1:
            true_str = "right"
            right_accs.append(acc_mean)
        elif true_class == 2:
            true_str = "rest"
            rest_accs.append(acc_mean)
        else:
            true_str = "unknown"
        print(f"{file}: True={true_str}, Left={left_count}, Right={right_count}, Rest={rest_count}, Total={total}, AccMean={acc_mean:.2%}, AccStd={acc_std:.2%}")
        all_total += total
        all_left += left_count
        all_right += right_count
        all_rest += rest_count
        all_accs.append(acc_mean)

    if all_total > 0:
        print(f"\nTotal: Left={all_left}, Right={all_right}, Rest={all_rest}")
        print(f"Mean accuracy across files: {np.mean(all_accs):.2%}, Std: {np.std(all_accs):.2%}")
        if left_accs:
            print(f"Left  Mean: {np.mean(left_accs):.2%}, Std: {np.std(left_accs):.2%}")
        if right_accs:
            print(f"Right Mean: {np.mean(right_accs):.2%}, Std: {np.std(right_accs):.2%}")
        if rest_accs:
            print(f"Rest  Mean: {np.mean(rest_accs):.2%}, Std: {np.std(rest_accs):.2%}")
    else:
        print("\nNo files were evaluated.")
