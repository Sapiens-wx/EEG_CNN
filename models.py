MODEL_ALIASES = {
    'cnn': ['cnn'],
    'cnn_featureExtraction': ['cnn_featureExtraction','cnn_featExtract'],
    'transformer': ['transformer'],
    'cnn_lstm': ['cnn+lstm', 'cnnlstm', 'cnn_lstm'],
    'dual_attention_transformer': ['davit', 'dualattentiontransformer', 'dual_attention_transformer'],
    'hybridcnn': ['hybridcnn', 'hybrid_cnn', 'cnn+spectral']
}

def get_standard_model_name(name):
    name = name.strip().replace(' ', '_').lower()
    for standard, aliases in MODEL_ALIASES.items():
        if name == standard or name in [alias.lower() for alias in aliases]:
            return standard
    return None

def validate_model_name(name):
    return get_standard_model_name(name) is not None


def LoadModel(model_type, numSamples, numChannels, num_classes, model_optimizer='adam', windowSize=None):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    standard_name = get_standard_model_name(model_type)
    if standard_name is None:
        raise ValueError(f"Unknown model type: {model_type}")

    match standard_name:
        case 'cnn':
            return CNN(numSamples, numChannels, num_classes, model_optimizer)
        case 'cnn_featExtract':
            return CNN_featExtract(numSamples, numChannels, num_classes, model_optimizer)
        case 'transformer':
            return Transformer(numSamples, numChannels, num_classes, model_optimizer)
        case 'cnn_lstm':
            return CNNLSTM(numSamples, numChannels, num_classes, model_optimizer)
        case 'dual_attention_transformer':
            return DualAttentionTransformer(numSamples, numChannels, num_classes, model_optimizer)
        case 'hybridcnn':
            if windowSize is None:
                raise ValueError("windowSize must be provided for HybridCNN")
            return HybridCNN(windowSize, num_classes, model_optimizer=model_optimizer)
        case _:
            raise ValueError(f"Unknown model type: {model_type}")


def CNN(numSamples, numChannels, num_classees, model_optimizer = 'adam'):
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers

    model = models.Sequential([
        layers.Input(shape=(numSamples, numChannels)),
        # CNN
        layers.Conv1D(filters=8, kernel_size=32, strides=1, padding='same',kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(filters=16, kernel_size=16, strides=1, padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(pool_size=2),

        layers.Flatten(),
        layers.Dense(8, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.3),

        # output
        layers.Dense(num_classees, activation='softmax'),
    ])
    model.compile(optimizer = model_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def CNN_featExtract(numSamples, numChannels, num_classes, model_optimizer = 'adam'):
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers

    model = models.Sequential([
        layers.Input(shape=(numSamples, numChannels)),  # 4 channels: TP9, AF7, AF8, TP10

        # 频域特征增强层
        layers.Conv1D(
            filters=16, 
            kernel_size=5,               # 小卷积核捕捉局部频域模式
            strides=1,
            padding='same',
            kernel_regularizer=regularizers.l2(1e-4),
            activation='relu'           # 直接整合Activation简化结构
        ),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),  # 降采样至 (18, 16)

        # 深层特征提取
        layers.Conv1D(
            filters=32, 
            kernel_size=3,               # 更小的卷积核细化特征
            strides=1,
            padding='same',
            kernel_regularizer=regularizers.l2(1e-4),
            activation='relu'
        ),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),  # 降采样至 (9, 32)

        # 全局特征聚合
        layers.GlobalAveragePooling1D(), # 替代Flatten + Dense，减少参数量
        # 分类头
        layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer = model_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def Transformer(numSamples, numChannels, num_classees, model_optimizer = 'adam'):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    inputs = layers.Input(shape=(numSamples, numChannels))  # 4 channels: TP9, AF7, AF8, TP10
    x = layers.Dense(64)(inputs)
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
    outputs = layers.Dense(num_classees, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer= model_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def CNNLSTM(numSamples, numChannels, num_classees, model_optimizer = 'adam'):
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers

    model = models.Sequential([
        layers.Input(shape=(numSamples, numChannels)),  # 4 channels: TP9, AF7, AF8, TP10
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
        layers.Dense(num_classees, activation='softmax'),

    ])
    model.compile(optimizer=model_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def DualAttentionTransformer(numSamples, numChannels, num_classes, model_optimizer = 'adam'):
    import tensorflow as tf
    from tensorflow.keras import layers, models
    inputs = layers.Input(shape=(numSamples, numChannels))  # 4 channels: TP9, AF7, AF8, TP10

    # Channel Attention
    channel_avg = layers.GlobalAveragePooling1D()(inputs)
    channel_max = layers.GlobalMaxPooling1D()(inputs)
    channel_concat = layers.Concatenate()([channel_avg, channel_max])
    channel_dense = layers.Dense(8, activation='relu')(channel_concat)
    channel_attention = layers.Dense(4, activation='sigmoid')(channel_dense)
    channel_attention = layers.Reshape((1, 4))(channel_attention)
    channel_attended = layers.Multiply()([inputs, channel_attention])

    # Temporal Attention
    temporal_dense = layers.Dense(32, activation='relu')(channel_attended)
    temporal_attention = layers.Dense(windowSize, activation='softmax')(temporal_dense)
    temporal_attention = layers.Reshape((windowSize, 1))(temporal_attention)
    temporal_attended = layers.Multiply()([channel_attended, temporal_attention])

    # Transformer Encoder
    x = layers.Dense(64)(temporal_attended)
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
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=model_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def HybridCNN(windowSize, num_classes, freq_bins=33, time_bins=5, model_optimizer="adam"):
    import tensorflow as tf
    from tensorflow.keras import layers, models, Input
    from tensorflow.keras.models import Model

    # --- Branch A: 1D CNN on raw waveform ---
    input_raw = Input(shape=(windowSize, 4), name="raw_input")
    x1 = layers.Conv1D(8, kernel_size=32, padding='same', activation='relu')(input_raw)
    x1 = layers.MaxPooling1D(pool_size=2)(x1)
    x1 = layers.Conv1D(16, kernel_size=16, padding='same', activation='relu')(x1)
    x1 = layers.MaxPooling1D(pool_size=2)(x1)
    x1 = layers.GlobalAveragePooling1D()(x1)  # ✅ 用 GAP 替代 Flatten

    # --- Branch B: 2D CNN on SFFT + DWT ---
    input_freq = Input(shape=(8, freq_bins, time_bins), name="spectral_input")
    x2 = layers.Permute((2, 3, 1))(input_freq)  # → shape: (freq, time, channels)
    x2 = layers.Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu')(x2)
    x2 = layers.MaxPooling2D(pool_size=(2, 2))(x2)
    x2 = layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(x2)
    x2 = layers.MaxPooling2D(pool_size=(2, 2))(x2)
    x2 = layers.GlobalAveragePooling2D()(x2)  # ✅ 用 GAP 替代 Flatten

    # --- Merge and output ---
    merged = layers.Concatenate()([x1, x2])
    merged = layers.Dense(16, activation='relu')(merged)  # ✅ 降维 Dense
    merged = layers.Dropout(0.5)(merged)
    output = layers.Dense(num_classes, activation='softmax')(merged)

    model = Model(inputs=[input_raw, input_freq], outputs=output)
    model.compile(optimizer=model_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

