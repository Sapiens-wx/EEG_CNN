import os
import numpy as np
import models
import tensorflow as tf
from labels import validate_labels

"""
This script supports finetuning EEG classifiers using newly labeled data.

Model Support:
- Works with all architectures defined in models.py (CNN, Transformer, etc.)
- To extend with custom models, add a function in models.py (e.g. create_my_model),
  then register the name in the MODEL_ALIASES and update LoadModel logic.

Usage:
- Called by realtime_preprocess_and_extract.py after unsupervised labeling
- New data is expected to be a .npy dict: {'segments': np.ndarray, 'labels': np.ndarray}
"""


def finetune_model_on_new_data(
    model_path,
    new_data_path,
    model_type="CNN",
    epochs=5,
    freeze_all_but_last=True
):
    """
    Finetune an existing model on new labeled EEG segments.

    Args:
        model_path (str): Path to saved .keras model
        new_data_path (str): Path to new .npy file containing {'segments', 'labels'}
        model_type (str): Model architecture type (CNN, Transformer, etc.)
        epochs (int): Number of fine-tuning epochs
        freeze_all_but_last (bool): Whether to freeze all layers except the last dense layer
    """
    # Load new training data
    data = np.load(new_data_path, allow_pickle=True).item()
    segments = data["segments"]
    labels = data["labels"]
    num_classes = labels.shape[1]
    window_size = segments.shape[1]

    # Load model architecture & weights
    model = models.LoadModel(model_type, window_size, num_classes)
    model.load_weights(model_path)
    print(f"[INFO] Loaded model from: {model_path}")

    # Optionally freeze all layers except last Dense
    if freeze_all_but_last:
        for layer in model.layers[:-1]:
            layer.trainable = False
        print("[INFO] All layers frozen except final Dense output layer")

    # Compile again after changing trainability
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train
    model.fit(segments, labels, epochs=epochs, batch_size=32)

    # Evaluate on new data
    loss, acc = model.evaluate(segments, labels, verbose=0)
    print(f"[INFO] Accuracy after fine-tuning on new data: {acc:.4f}")


    # Save updated model
    basename = os.path.basename(model_path).replace(".keras", "")
    time_str = tf.timestamp().numpy().astype(int)
    save_name = f"{basename}_finetuned_{time_str}.keras"
    save_path = os.path.join(os.path.dirname(model_path), save_name)
    model.save(save_path)
    print(f"[INFO] Fine-tuned model saved to: {save_path}")
