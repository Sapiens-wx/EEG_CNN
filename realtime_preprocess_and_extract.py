import os
import numpy as np
from datetime import datetime
from new_preprocessing import preprocess_eeg_csv
from clustering import cluster_segments_pca_kmeans  # You just created this

def get_latest_csv(folder="recorded_data"):
    """Find the most recently modified EEG CSV file."""
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder '{folder}' not found.")
    
    csv_files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in folder '{folder}'.")
    
    csv_files.sort(key=lambda f: os.path.getmtime(os.path.join(folder, f)), reverse=True)
    return os.path.join(folder, csv_files[0])

def preprocess_latest_recorded_data():
    """Find the latest EEG CSV file and preprocess it into EEG segments."""
    try:
        latest_file = get_latest_csv()
        print(f"[INFO] Preprocessing latest file: {latest_file}")
        segments = preprocess_eeg_csv(latest_file)
        print(f"[INFO] EEG segments shape: {segments.shape}")
        return segments
    except Exception as e:
        print(f"[ERROR] {e}")
        return None

def get_preprocessed_and_clustered_segments():
    """
    Preprocess the latest EEG recording and cluster the segments into 3 groups.
    This function is intended to be called by the game after each round.

    Returns:
        segments (np.ndarray): EEG segments, shape (n, window_size, 4)
        cluster_labels (np.ndarray): Cluster assignments, shape (n,)
    """
    segments = preprocess_latest_recorded_data()
    if segments is None:
        return None, None
    cluster_labels, _ = cluster_segments_pca_kmeans(segments)
    return segments, cluster_labels

if __name__ == "__main__":
    segments, labels = get_preprocessed_and_clustered_segments()
    if segments is not None and labels is not None:
        import collections
        print("[INFO] Cluster distribution:", collections.Counter(labels))
        for i in range(3):
            indices = np.where(labels == i)[0]
            if len(indices) == 0:
                print(f"[INFO] Cluster {i} is empty.")
                continue
            example = segments[indices[0]]
            print(f"[INFO] Cluster {i} example: mean={np.mean(example):.2f}, std={np.std(example):.2f}")
