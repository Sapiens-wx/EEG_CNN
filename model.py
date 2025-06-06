import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers.legacy import Adam

def LoadModel(path):
    if os.path.exists(path):
        model=tf.keras.models.load_model(path);
    else:
        model=CreateModel();
    return model;

def CreateModel():
    model=models.Sequential([
        # input shape: 256, 5 channels
        # CNN
        layers.Conv1D(filters=16, kernel_size=5, strides=1, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(filters=32, kernel_size=3, strides=1, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(pool_size=2),
        # LSTM
        #layers.Reshape((64,32)),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(32),

        layers.Dense(16, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(3, activation='softmax'),
    ])
    model.compile(optimizer='adam',loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
