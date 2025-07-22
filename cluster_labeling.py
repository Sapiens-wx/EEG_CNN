import os
import numpy as np
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from labels import extract_labels_from_filename

def load_historical_class_centers(preprocessed_dir="preprocessed_data", label_list=("left", "right", "neutral")):
    """
    Load previously labeled EEG segments and compute the average feature vector per class.

    Args:
        preprocessed_dir (str): Directory containing .npy preprocessed EEG files
        label_list (tuple): Set of supported class labels to consider

    Returns:
        dict: {label_name: average_feature_vector}
    """
    vectors = {label: [] for label in label_list}

    for fname in os.listdir(preprocessed_dir):
        if not fname.endswith(".npy"):
            continue
        label_info = extract_labels_from_filename(fname)
        labels = [list(d.keys())[0] for d in label_info]
        if not all(l in label_list for l in labels):
            continue
        data = np.load(os.path.join(preprocessed_dir, fname), allow_pickle=True).item()
        segments = data["segments"]
        onehot_labels = data["labels"]
        for seg, onehot in zip(segments, onehot_labels):
            idx = np.argmax(onehot)
            label = label_list[idx]
            vectors[label].append(seg.reshape(-1))  # Flatten

    centers = {}
    for label in vectors:
        if len(vectors[label]) > 0:
            centers[label] = np.mean(np.stack(vectors[label]), axis=0)
    return centers

def assign_cluster_labels_by_similarity(segments, cluster_labels, historical_centers):
    """
    Assign real-world labels (e.g. left/right/neutral) to unsupervised clusters
    by comparing their centers to historical class centers using cosine similarity.

    Args:
        segments (np.ndarray): EEG segments (n, window_size, 4)
        cluster_labels (np.ndarray): Cluster assignments for each segment
        historical_centers (dict): Dictionary of historical label -> feature center

    Returns:
        dict: {cluster_index: assigned_label}
    """
    num_clusters = len(set(cluster_labels))
    cluster_centers = []

    # Compute the center of each cluster
    for i in range(num_clusters):
        indices = np.where(cluster_labels == i)[0]
        cluster_data = segments[indices]
        flattened = cluster_data.reshape(cluster_data.shape[0], -1)
        cluster_centers.append(np.mean(flattened, axis=0))

    # Match each cluster center to the most similar historical class
    mapping = {}
    used_labels = set()
    for i, c_center in enumerate(cluster_centers):
        best_label = None
        best_score = -1
        for label, h_center in historical_centers.items():
            if label in used_labels:
                continue
            sim = cosine_similarity([c_center], [h_center])[0][0]
            if sim > best_score:
                best_score = sim
                best_label = label
        mapping[i] = best_label
        used_labels.add(best_label)

    return mapping

def save_labeled_segments(segments, cluster_labels, cluster_to_label_map, outdir="preprocessed_data"):
    """
    Save clustered EEG segments with their assigned labels to a .npy file.

    Args:
        segments (np.ndarray): EEG segments, shape (n, window_size, 4)
        cluster_labels (np.ndarray): Cluster assignments (n,)
        cluster_to_label_map (dict): Mapping from cluster index to label name
        outdir (str): Output directory to save the labeled file
    """
    os.makedirs(outdir, exist_ok=True)

    # Convert cluster index to label name
    label_names = [cluster_to_label_map[c] for c in cluster_labels]

    # One-hot encode labels
    label_set = sorted(set(label_names))
    label_index = {name: i for i, name in enumerate(label_set)}
    onehot_labels = np.zeros((len(label_names), len(label_set)))
    for i, name in enumerate(label_names):
        onehot_labels[i][label_index[name]] = 1

    data = {
        "segments": segments,
        "labels": onehot_labels
    }

    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"preprocessed_auto_{'-'.join(label_set)}_{time_str}.npy"
    path = os.path.join(outdir, filename)
    np.save(path, data)
    print(f"[INFO] Saved labeled segments to {path}")
