import sys
import numpy as np
import os
import model_switchable as model
from sklearn.model_selection import train_test_split
import tensorflow as tf

# 解析命令行参数
model_type = "cnn"  # 默认
epochs = 100        # 默认
print_result = False
for i, arg in enumerate(sys.argv):
    if arg in ("-model", "--model") and i + 1 < len(sys.argv):
        model_type = sys.argv[i + 1].lower().replace(' ', '')
    if arg in ("--epoch", "-epoch", "-epochs") and i + 1 < len(sys.argv):
        try:
            epochs = int(sys.argv[i + 1])
        except ValueError:
            print("Invalid epoch value, using default 100.")
    if arg == "-printResult":
        print_result = True

print(f"Selected model type: {model_type}")
print(f"Training epochs: {epochs}")

# 加载数据
segments = np.load(os.path.join("training_data", "preprocessed", "eeg_segments.npy"))
labels = np.load(os.path.join("training_data", "preprocessed", "eeg_labels.npy"))

x_train, x_test, y_train, y_test = train_test_split(segments, labels, test_size=0.2, random_state=42)

if y_train.shape[-1] != 9:
    raise ValueError(f"y_train shape[-1]={y_train.shape[-1]}, should be 9 (9-class one-hot).")

# 根据参数选择模型
if model_type == "cnn":
    net = model.LoadCNNModel()
    model_name = "model_cnn.keras"
elif model_type == "transformer":
    net = model.LoadTransformerModel()
    model_name = "model_transformer.keras"
elif model_type in ("cnn+lstm", "cnnlstm", "cnn_lstm"):
    net = model.LoadCNNLSTMModel()
    model_name = "model_cnnlstm.keras"
else:
    raise ValueError(f"Unknown model type: {model_type}")

print("GPU Available:", tf.config.list_physical_devices('GPU'))

# 训练
history = net.fit(
    x_train, y_train,
    epochs=epochs,
    batch_size=32,
    validation_data=(x_test, y_test)
)

if print_result:
    print("\nModel Summary:")
    net.summary()

loss, accuracy = net.evaluate(x_test, y_test)
print(f"loss={loss}, accuracy={accuracy}")
net.save(model_name)
input("press enter to exit ...")