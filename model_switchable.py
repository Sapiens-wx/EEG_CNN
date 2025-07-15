import tensorflow as tf
from tensorflow.keras import layers, models

def LoadCNNModel():
    # Example CNN for (256, 5) input
    model = models.Sequential([
        layers.Input(shape=(256, 5)),
        layers.Conv1D(32, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(9, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def LoadTransformerModel():
    # Example Transformer for (256, 5) input
    inputs = layers.Input(shape=(256, 5))
    x = layers.Dense(64)(inputs)
    for _ in range(2):
        attn_output = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
        x = layers.Add()([x, attn_output])
        x = layers.LayerNormalization()(x)
        ff = layers.Dense(128, activation='relu')(x)
        ff = layers.Dense(64)(ff)
        x = layers.Add()([x, ff])
        x = layers.LayerNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(9, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def LoadCNNLSTMModel():
    # Example CNN+LSTM for (256, 5) input
    model = models.Sequential([
        layers.Input(shape=(256, 5)),
        layers.Conv1D(32, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.LSTM(32, return_sequences=False),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(9, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
