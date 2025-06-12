import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout, Conv1D, Conv2D, Flatten, Reshape, MaxPooling1D, MaxPooling2D, AveragePooling2D, BatchNormalization, Permute, ReLU, Softmax
from tensorflow.keras.optimizers.legacy import Adam
from sklearn.model_selection import train_test_split
import model
import numpy as np
import os

# Load preprocessed data
segments = np.load(os.path.join("training_data\preprocessed\eeg_segments.npy"))  # CHANGE THIS PATH
labels = np.load(os.path.join("training_data\preprocessed\eeg_labels.npy"))  # CHANGE THIS PATH

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(segments, labels, test_size=0.2, random_state=42)

# Print data shapes for verification
print(f"X_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")

# 检查标签shape是否为9类one-hot
if y_train.shape[-1] != 9:
    raise ValueError(f"y_train shape[-1]={y_train.shape[-1]}, 但应为9（9类one-hot标签）。请检查预处理脚本和数据！")

# Initialize CNN model
model_name="model.keras";
cnn = model.LoadModel(model_name);

# Training loop
if __name__ == "__main__":
    """
    Main script for training the EEG Transformer model.
    """
    # check if gpu is available
    print("GPU Available:", tf.config.list_physical_devices('GPU'))
    # train model
    history=cnn.fit(x_train, y_train,
                    epochs= 300,
                    batch_size=32,
                    validation_data=(x_test, y_test)
    )

    loss, accuracy=cnn.evaluate(x_test, y_test)
    print(f"loss={loss}, accuracy={accuracy}")
    cnn.save(model_name)
    input("press enter to exit ...")
