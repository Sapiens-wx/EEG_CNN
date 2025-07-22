import os
import numpy as np
import argparse
from datetime import datetime
from new_preprocessing import preprocess_eeg_csv
from clustering import cluster_segments_pca_kmeans
from cluster_labeling import (
    load_historical_class_centers,
    assign_cluster_labels_by_similarity,
    save_labeled_segments,
)
from model_finetune import finetune_model_on_new_data

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
    """Preprocess the most recent EEG CSV file into segments."""
    try:
        latest_file = get_latest_csv()
        print(f"[INFO] Preprocessing latest file: {latest_file}")
        segments = preprocess_eeg_csv(latest_file)
        print(f"[INFO] EEG segments shape: {segments.shape}")
        return segments
    except Exception as e:
        print(f"[ERROR] {e}")
        return None

def run_clustering_on_segments(segments):
    """Run KMeans clustering on EEG segments."""
    cluster_labels, _ = cluster_segments_pca_kmeans(segments)
    return cluster_labels

def assign_labels_and_save(segments, cluster_labels):
    """Assign labels to clusters and save the labeled data for future use.

    Returns:
        str: Path to the saved labeled .npy file
    """
    historical_centers = load_historical_class_centers()
    cluster_to_label_map = assign_cluster_labels_by_similarity(segments, cluster_labels, historical_centers)
    path = save_labeled_segments(segments, cluster_labels, cluster_to_label_map)
    return path

def find_latest_model(model_dir="models"):
    """Find the most recently saved .keras model."""
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".keras")]
    if not model_files:
        print("[WARN] No models found.")
        return None
    model_files.sort(key=lambda f: os.path.getmtime(os.path.join(model_dir, f)), reverse=True)
    return os.path.join(model_dir, model_files[0])

def run_full_adaptive_update():
    """
    Main function to be called after each game round.
    It performs:
    - EEG preprocessing
    - Unsupervised clustering
    - Cluster-to-label assignment
    - Labeled data saving
    - Optional model fine-tuning
    """
    segments = preprocess_latest_recorded_data()
    if segments is None:
        print("[ERROR] No EEG segments to process.")
        return

    cluster_labels = run_clustering_on_segments(segments)
    new_data_path = assign_labels_and_save(segments, cluster_labels)

    model_path = find_latest_model()
    if model_path and new_data_path:
        print("[INFO] Starting fine-tuning...")
        finetune_model_on_new_data(
            model_path=model_path,
            new_data_path=new_data_path,
            model_type="CNN",  # Update this if using another model
            epochs=5
        )
    else:
        print("[WARN] Skipping fine-tuning due to missing model or data.")

    print("[INFO] Adaptive update completed.")

def main():
    parser = argparse.ArgumentParser(description="Run EEG adaptive update pipeline.")
    parser.add_argument("-model", type=str, default="CNN", help="Model type to load for finetuning (default: CNN)")
    parser.add_argument("-epochs", type=int, default=5, help="Epochs for fine-tuning (default: 5)")
    parser.add_argument("--noFinetune", action="store_true", help="Skip model fine-tuning step")
    args = parser.parse_args()

    segments = preprocess_latest_recorded_data()
    if segments is None:
        print("[ERROR] No EEG segments to process.")
        return

    cluster_labels = run_clustering_on_segments(segments)
    new_data_path = assign_labels_and_save(segments, cluster_labels)

    if args.noFinetune:
        print("[INFO] Skipping fine-tuning (as requested).")
        return

    model_path = find_latest_model()
    if model_path and new_data_path:
        print("[INFO] Starting fine-tuning...")
        finetune_model_on_new_data(
            model_path=model_path,
            new_data_path=new_data_path,
            model_type=args.model,
            epochs=args.epochs
        )
    else:
        print("[WARN] Skipping fine-tuning due to missing model or data.")
    print("[INFO] Adaptive update completed.")

if __name__ == "__main__":
    main()
