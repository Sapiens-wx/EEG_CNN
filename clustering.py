from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

def cluster_segments_pca_kmeans(segments, n_clusters=3):
    """
    Flatten EEG segments, reduce dimensionality with PCA, then cluster using KMeans.

    Args:
        segments (np.ndarray): EEG segments with shape (num_samples, window_size, 4)
        n_clusters (int): Number of clusters to generate (default: 3)

    Returns:
        cluster_labels (np.ndarray): Cluster assignment for each segment (shape: num_samples)
        cluster_centers (np.ndarray): Cluster centers in PCA-reduced space
    """
    num_samples, window_size, num_channels = segments.shape

    # Flatten each segment into a 1D vector: (samples, window_size * channels)
    flattened = segments.reshape(num_samples, window_size * num_channels)

    # Apply PCA to reduce to 20 dimensions
    pca = PCA(n_components=20)
    reduced = pca.fit_transform(flattened)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(reduced)

    return cluster_labels, kmeans.cluster_centers_
