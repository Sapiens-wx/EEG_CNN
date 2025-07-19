MODEL_ALIASES = {
    'cnn': ['cnn'],
    'transformer': ['transformer'],
    'cnn_lstm': ['cnn+lstm', 'cnnlstm', 'cnn_lstm'],
    'dual_attention_transformer': ['davit', 'dualattentiontransformer', 'dual_attention_transformer']
}

def get_standard_model_name(name):
    name = name.strip().replace(' ', '_').lower()
    for standard, aliases in MODEL_ALIASES.items():
        if name == standard or name in [alias.lower() for alias in aliases]:
            return standard
    return None

def validate_model_name(name):
    return get_standard_model_name(name) is not None


def LoadModel(model_type, windowSize, num_classes, model_optimizer='adam'):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    standard_name = get_standard_model_name(model_type)
    if standard_name is None:
        raise ValueError(f"Unknown model type: {model_type}")

    match standard_name:
        case 'cnn':
            return CNN(windowSize, num_classes, model_optimizer)
        case 'transformer':
            return Transformer(windowSize, num_classes, model_optimizer)
        case 'cnn_lstm':
            return CNNLSTM(windowSize, num_classes, model_optimizer)
        case 'dual_attention_transformer':
            return DualAttentionTransformer(windowSize, num_classes, model_optimizer)
        case _:
            raise ValueError(f"Unknown model type: {model_type}")


def CNN(windowSize, num_classees, model_optimizer = 'adam'):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Input(shape=(windowSize, 4)), # 4 channels: TP9, AF7, AF8, TP10
        layers.Conv1D(32, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classees, activation='softmax')
    ])
    model.compile(optimizer = model_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def Transformer(windowSize, num_classees, model_optimizer = 'adam'):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    inputs = layers.Input(shape=(windowSize, 4))  # 4 channels: TP9, AF7, AF8, TP10
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

def CNNLSTM(windowSize, num_classees, model_optimizer = 'adam'):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Input(shape=(windowSize, 4)),  # 4 channels: TP9, AF7, AF8, TP10
        layers.Conv1D(32, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.LSTM(32, return_sequences=False),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classees, activation='softmax')
    ])
    model.compile(optimizer=model_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def DualAttentionTransformer(windowSize, num_classes, model_optimizer = 'adam'):
    import tensorflow as tf
    from tensorflow.keras import layers, models
    inputs = layers.Input(shape=(windowSize, 4))  # 4 channels: TP9, AF7, AF8, TP10

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