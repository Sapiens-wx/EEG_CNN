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
    # 假设输入 shape: (256, 4) 或 (256, 5)，如有不同请调整
    input_shape = (256, 4)
    inputs = layers.Input(shape=input_shape)
    # 线性投影到d_model
    x = layers.Dense(64)(inputs)
    # Transformer Encoder Block
    for _ in range(2):
        # Multi-head Self Attention
        attn_output = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
        x = layers.Add()([x, attn_output])
        x = layers.LayerNormalization()(x)
        # Feed Forward
        ff = layers.Dense(128, activation='relu')(x)
        ff = layers.Dense(64)(ff)
        x = layers.Add()([x, ff])
        x = layers.LayerNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(9, activation='softmax')(x)  # 9类输出
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
