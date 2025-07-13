import os
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers.legacy import Adam
import config

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
        layers.Conv1D(filters=8, kernel_size=32, strides=1, padding='same',kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(filters=16, kernel_size=16, strides=1, padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(pool_size=2),
        # LSTM
        #layers.Reshape((64,32)),
        layers.LSTM(32, return_sequences=True, dropout=0.3, recurrent_dropout=0.2, kernel_regularizer=regularizers.l2(1e-4)),
        layers.LSTM(16, dropout=0.3, recurrent_dropout=0.2, kernel_regularizer=regularizers.l2(1e-4), recurrent_regularizer=regularizers.l2(1e-4)),

        layers.Dense(8, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.3),

        # output
        layers.Dense(len(config.recordEEG.commands), activation='softmax'),
    ])
    model.compile(optimizer='adam',loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
