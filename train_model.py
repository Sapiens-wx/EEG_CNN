import argparse
import os
from labels import validate_labels, format_valid_labels_message
import glob
import models
from eeg_features import extract_sfft_and_dwt

parser = argparse.ArgumentParser(description="EEG Classification Model Training Script")
parser.add_argument("-model", type=str, required=True, help="Model type: CNN, Transformer, CNN+LSTM, DaViT. Read models.py")
parser.add_argument("-epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("-windowSize", type=int, default=256, help="Window size for EEG segments")
parser.add_argument("-slidingWindow", type=int, default=128, help="Sliding window step for EEG segments")
parser.add_argument("-labels", type=str, required=True, help="Comma separated labels for EEG data")
parser.add_argument("-trainDataRatio", type=float, default=0.8, nargs='?', help="Ratio of training data to total data (default: 0.8). The rest will be used for testing.")
parser.add_argument("-preprocessedFilePath", type=str, help="Path to a preprocessed .npy file. If provided, skips labels recognition logic.")
args = parser.parse_args()

if args.epochs <= 0:
    print("Error: epochs must be greater than 0.")
    exit(1)
if args.windowSize <= 0:
    print("Error: windowSize must be greater than 0.")
    exit(1)
if args.slidingWindow <= 0 or args.slidingWindow > args.windowSize:
    print("Error: slidingWindow must be greater than 0 and less than windowSize.")
    exit(1)

if not (0 < args.trainDataRatio <= 1):
    print("Error: trainDataRatio must be between 0 (exclusive) and 1 (inclusive).")
    exit(1)

# Check if preprocessedFilePath is provided
if args.preprocessedFilePath:
    if not os.path.isfile(args.preprocessedFilePath):
        print(f"Error: The file '{args.preprocessedFilePath}' does not exist.")
        exit(1)
    selected_file = args.preprocessedFilePath
    # 验证labels
    labels_list, _, invalid_labels = validate_labels(args.labels)
    if not labels_list:
        print("Error: Labels cannot be empty.")
        exit(1)
    if invalid_labels:
        print(f"Error: Invalid labels: {', '.join(invalid_labels)}")
        print(format_valid_labels_message())
        exit(1)
    print(f"Using provided preprocessed file: {os.path.basename(selected_file)}")
else:
    # 验证labels
    labels_list, _, invalid_labels = validate_labels(args.labels)
    if not labels_list:
        print("Error: Labels cannot be empty.")
        exit(1)
    if invalid_labels:
        print(f"Error: Invalid labels: {', '.join(invalid_labels)}")
        print(format_valid_labels_message())
        exit(1)

    # 参数和模型类型验证提前，避免不必要的包导入
    import models
    if (models.validate_model_name(args.model) is False):
        print(f"Error: Invalid model type '{args.model}'. Valid options are: CNN, Transformer, CNN+LSTM, DaViT.")
        exit(1)

    # 检查 preprocessed 目录下是否有符合 label 的 numpy 文件（文件名包含所有 label abbr）
    preprocessed_dir = "preprocessed_data"
    matched_files = []
    abbrs = [abbr for abbr in labels_list]
    if not os.path.exists(preprocessed_dir):
        print("Preprocessed folder does not exist.")
        exit(1)
    for fname in os.listdir(preprocessed_dir):
        if fname.endswith('.npy') and all(abbr in fname for abbr in abbrs):
            matched_files.append(os.path.join(preprocessed_dir, fname))
    if not matched_files:
        print(f"No preprocessed .npy files found in '{preprocessed_dir}' matching labels: {', '.join(abbrs)}")
        # 检查 recorded_data 目录下是否有足够的 csv 文件
        recorded_dir = "recorded_data"
        if not os.path.exists(recorded_dir):
            print(f"No preprocessed .npy and recorded_data folder does not exist.")
            exit(1)
        csv_files = [fname for fname in os.listdir(recorded_dir) if fname.endswith('.csv')]
        # 检查每个label至少有一个csv文件（忽略大小写）
        label_csv_map = {abbr: [] for abbr in abbrs}
        for fname in csv_files:
            fname_lower = fname.lower()
            for abbr in abbrs:
                if abbr.lower() in fname_lower:
                    label_csv_map[abbr].append(fname)
        missing_labels = [abbr for abbr, files in label_csv_map.items() if len(files) == 0]
        if missing_labels:
            print(f"No csv files found for label(s): {', '.join(missing_labels)} in '{recorded_dir}'")
            exit(1)
        print(f"Found csv files for all required labels in '{recorded_dir}'.")
        user_input = input("Do you want to preprocess these csv files now and continue training? (y/n): ")
        if user_input.strip().lower() == 'y':
            # 调用预处理脚本
            import subprocess
            preprocess_script = os.path.join(os.path.dirname(__file__), 'preprocess_eeg.py')
            if not os.path.exists(preprocess_script):
                print("preprocess_eeg.py not found!")
                exit(1)
            print("Running preprocessing...")
            result = subprocess.run(["python", preprocess_script, "-labels", ','.join(abbrs), '-slidingWindow', str(args.slidingWindow) ,'-windowSize', str(args.windowSize)], cwd=os.path.dirname(__file__))
            if result.returncode != 0:
                print("Preprocessing failed.")
                exit(1)
            # 重新查找npy文件
            matched_files = []
            for fname in os.listdir(preprocessed_dir):
                if fname.endswith('.npy') and all(abbr in fname for abbr in abbrs):
                    matched_files.append(os.path.join(preprocessed_dir, fname))
            if not matched_files:
                print("Preprocessing finished but no suitable .npy file was generated.")
                exit(1)
            matched_files.sort(key=lambda x: os.path.getctime(x), reverse=True)
            selected_file = matched_files[0]
            print(f"Using preprocessed file: {os.path.basename(selected_file)}")
        else:
            print("Aborted by user.")
            exit(1)
    matched_files.sort(key=lambda x: os.path.getctime(x), reverse=True)
    selected_file = matched_files[0]
    print(f"Using preprocessed file: {os.path.basename(selected_file)}")


# 加载 npy 文件，只处理单个 dict/tuple npy 文件
import numpy as np
segments = None
labels = None
try:
    data = np.load(selected_file, allow_pickle=True)
    print(f"Loaded data shape: {getattr(data, 'shape', None)}")
    print(f"Type of loaded data: {type(data)}")
    # 兼容 dict 保存方案
    if isinstance(data, np.ndarray):
        try:
            data = data.item()
            print(f"Loaded dict from np.ndarray, keys: {list(data.keys())}")
        except Exception as e:
            print(f"Error converting np.ndarray to dict: {e}")
            exit(1)
    if isinstance(data, dict):
        segments = data.get('segments', None)
        labels = data.get('labels', None)
        print(f"Keys in dict: {list(data.keys())}")
    elif isinstance(data, tuple):
        print(f"Tuple length: {len(data)}")
        if len(data) == 2:
            segments, labels = data
    else:
        print(f"Error: npy file must be a dict or tuple containing segments and labels. Got type: {type(data)}")
        exit(1)
    if segments is None or labels is None:
        print("Error: Could not find both segments and labels in npy file.")
        exit(1)
except Exception as e:
    print(f"Error loading npy file: {e}")
    exit(1)


# 使用 sklearn 的 train_test_split 按 ratio 划分训练和测试集
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    segments, labels, test_size=1-args.trainDataRatio, random_state=42)

if args.model.lower() in ['hybridcnn', 'hybrid_cnn', 'cnn+spectral']:
    print("[INFO] Preparing dual-input features for HybridCNN...")
    # segments.shape: (N, 256, 4)
    x_raw_train = x_train
    x_raw_test = x_test

    x_spec_train = np.array([extract_sfft_and_dwt(seg) for seg in x_train])
    x_spec_test = np.array([extract_sfft_and_dwt(seg) for seg in x_test])

    x_train = [x_raw_train, x_spec_train]
    x_test = [x_raw_test, x_spec_test]

def check_data(dataset):
    assert isinstance(dataset, np.ndarray), "check_data expects a numpy array"
    if np.isnan(dataset).any():
        raise ValueError('NaN exists in dataset')
    if np.isinf(dataset).any():
        raise ValueError('Infinity exists in dataset')
    print("Data range: min={}, max={}".format(np.min(dataset), np.max(dataset)))


if isinstance(x_train, list):
    for i, x_part in enumerate(x_train):
        print(f"\n[INFO] Checking x_train[{i}]...")
        check_data(x_part)
else:
    check_data(x_train)

if isinstance(x_test, list):
    for i, x_part in enumerate(x_test):
        print(f"\n[INFO] Checking x_test[{i}]...")
        check_data(x_part)
else:
    check_data(x_test)



# 检查labels是否为one-hot编码
def is_one_hot(arr):
    return (
        isinstance(arr, np.ndarray)
        and arr.ndim == 2
        and np.all((arr == 0) | (arr == 1))
        and np.all(np.sum(arr, axis=1) == 1)
    )

if not is_one_hot(y_train) or not is_one_hot(y_test):
    print("Error: labels must be one-hot encoded.")
    exit(1)


if isinstance(x_train, list):  # HybridCNN
    numSamples = x_train[0].shape[1]
    numChannels = x_train[0].shape[2]
else:  # single input
    numSamples = x_train.shape[1]
    numChannels = x_train.shape[2]

model = models.LoadModel(
    model_type=args.model,
    numSamples=numSamples,
    numChannels=numChannels,
    num_classes=len(labels_list),
    model_optimizer='adam',
    windowSize=args.windowSize if args.model.lower() in ['hybridcnn', 'hybrid_cnn', 'cnn+spectral'] else None
)

# 根据参数选择模型
# model = models.LoadModel(
#     model_type=args.model,
#     numSamples=x_train.shape[1],
#     numChannels=x_train.shape[2],
#     num_classes=len(labels_list),
#     model_optimizer='adam'
# )

import tensorflow as tf

print("GPU Available:", tf.config.list_physical_devices('GPU'))

history = model.fit(
    x_train, y_train,
    epochs=args.epochs,
    batch_size=32,
    validation_data=(x_test, y_test)
)

# 训练后评估与保存模型
import datetime
save_dir = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(save_dir, exist_ok=True)

abbr_str = '-'.join([abbr for abbr in labels_list])
time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
model_type_str = args.model.lower().replace('+','').replace(' ','').replace('_','')
model_save_name = f"model_{model_type_str}_{abbr_str},{time_str}.keras"
model_save_path = os.path.join(save_dir, model_save_name)

loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

if hasattr(model, 'summary'):
    print("\nModel Summary:")
    model.summary()


print(f"Sample spectral shape: {x_spec_train[0].shape}")
print("[DEBUG] Spectral input stats: mean =", np.mean(x_spec_train), ", std =", np.std(x_spec_train))

input("Press Enter to exit ...")

