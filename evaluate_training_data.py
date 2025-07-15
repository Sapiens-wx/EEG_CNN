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

# 9-category definitions (统一label名称和顺序)
CATEGORY_NAMES = [
    "left", "right", "neutral",
    "left to right", "right to left",
    "left to neutral", "right to neutral",
    "neutral to left", "neutral to right"
]
CATEGORY_NAME_TO_IDX = {name: idx for idx, name in enumerate(CATEGORY_NAMES)}
CATEGORY_IDX_TO_NAME = {idx: name for idx, name in enumerate(CATEGORY_NAMES)}

def extract_label_from_filename(filename):
    lower = filename.lower().replace('-', ' ').replace('_', ' ')
    for name in CATEGORY_NAMES:
        name_key = name.lower().replace('-', ' ').replace('_', ' ')
        if name_key in lower:
            return CATEGORY_NAME_TO_IDX[name]
    for name in CATEGORY_NAMES:
        if name.split()[0] in lower:
            return CATEGORY_NAME_TO_IDX[name]
    return -1

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

def predict_file_with_path_count(file_path, window_size=256, test_count=10, filter_categories=None):
    import tensorflow as tf
    from preprocess_eeg import preprocess_data
    import pandas as pd
    import numpy as np
    df = pd.read_csv(file_path)
    data = df.drop(columns=['timestamps']).values
    all_window_predictions = []
    model = tf.keras.models.load_model('model.keras')
    file_base = os.path.basename(file_path)
    true_class = extract_label_from_filename(file_base)
    if filter_categories is not None and true_class not in filter_categories:
        return None
    n_windows = (len(data) - window_size) // window_size + 1 if len(data) >= window_size else 0
    for i in range(0, len(data) - window_size + 1, window_size):
        window = data[i:i + window_size]
        if len(window) == window_size:
            window_predictions = []
            preprocessed_window = preprocess_data(window)
            sequence = np.array(preprocessed_window)
            sequence = np.expand_dims(sequence, axis=0)
            for _ in range(test_count):
                outputs = model.predict(sequence, verbose=0)
                prediction = np.argmax(outputs[0])
                window_predictions.append(prediction)
            all_window_predictions.extend(window_predictions)
    # 统计
    counts = [all_window_predictions.count(i) for i in range(len(CATEGORY_NAMES))]
    total = len(all_window_predictions)
    correct = all_window_predictions.count(true_class) if true_class != -1 else 0
    acc = correct / total if total > 0 else 0
    return (file_base, true_class, counts, total, acc)

def predict_file_with_path_count_worker(file_path, window_size, test_count, idx, queue, filter_categories=None):
    import tensorflow as tf
    from preprocess_eeg import preprocess_data
    import pandas as pd
    import numpy as np
    df = pd.read_csv(file_path)
    data = df.drop(columns=['timestamps']).values
    n_windows = (len(data) - window_size) // window_size + 1 if len(data) >= window_size else 0
    model = tf.keras.models.load_model('model.keras')
    file_base = os.path.basename(file_path)
    true_class = extract_label_from_filename(file_base)
    if filter_categories is not None and true_class not in filter_categories:
        queue.put((idx, (file_base, true_class, [0]*len(CATEGORY_NAMES), 0, 0, 0, 0)))
        return
    acc_list = []
    total_counts = [0] * len(CATEGORY_NAMES)
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
        for i in range(len(CATEGORY_NAMES)):
            total_counts[i] += window_predictions.count(i)
    acc_mean = np.mean(acc_list)
    acc_std = np.std(acc_list)
    total = sum(total_counts)
    queue.put((idx, (file_base, true_class, total_counts, total, acc_mean, acc_std)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--jobs", type=int, default=1, help="Number of parallel processes")
    parser.add_argument("-testCount", type=int, default=5, help="Number of test repetitions per sample")
    parser.add_argument("-windowTime", type=int, default=500, help="Window size in milliseconds")
    parser.add_argument("-categories", nargs="*", default=None, help="Only evaluate specified categories (e.g. left right neutral ...)")
    args = parser.parse_args()

    if args.categories:
        filter_categories = [CATEGORY_NAME_TO_IDX[c] for c in args.categories if c in CATEGORY_NAME_TO_IDX]
    else:
        filter_categories = None

    training_data_dir = "training_data"
    files = [os.path.join(training_data_dir, f) for f in os.listdir(training_data_dir) if f.endswith(".csv")]
    results = []
    all_total = 0
    all_counts = [0] * len(CATEGORY_NAMES)
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
            pool.apply_async(predict_file_with_path_count_worker, (file, window_size, args.testCount, idx, queue, filter_categories))
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
        results = [r for r in file_results if r is not None]
    else:
        for file in files:
            print(f"\nProcessing {os.path.basename(file)}:")
            result = predict_file_with_path_count(file, window_size, args.testCount, filter_categories)
            if result is not None:
                results.append(result)

    print("\nSummary of predictions:")
    accs_by_class = [[] for _ in range(len(CATEGORY_NAMES))]
    for file, true_class, counts, total, acc_mean, *rest in results:
        true_str = CATEGORY_IDX_TO_NAME.get(true_class, "unknown")
        for i, c in enumerate(counts):
            all_counts[i] += c
        all_total += total
        if true_class >= 0 and true_class < len(CATEGORY_NAMES):
            accs_by_class[true_class].append(acc_mean)
        all_accs.append(acc_mean)
        count_str = ", ".join(f"{CATEGORY_IDX_TO_NAME[i]}={counts[i]}" for i in range(len(CATEGORY_NAMES)))
        print(f"{file}: True={true_str}, {count_str}, Total={total}, AccMean={acc_mean:.2%}")

    if all_total > 0:
        print(f"\nTotal: " + ", ".join(f"{CATEGORY_IDX_TO_NAME[i]}={all_counts[i]}" for i in range(len(CATEGORY_NAMES))))
        print(f"Mean accuracy across files: {np.mean(all_accs):.2%}, Std: {np.std(all_accs):.2%}")
        for i, name in CATEGORY_IDX_TO_NAME.items():
            if accs_by_class[i]:
                print(f"{name:15} Mean: {np.mean(accs_by_class[i]):.2%}, Std: {np.std(accs_by_class[i]):.2%}")
    else:
        print("\nNo files were evaluated.")
