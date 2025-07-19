# Recorded Data

This folder contains raw EEG data recorded from experiments or sessions.
Files are in .csv format and named according to the label and date of recording.

# For Preprocessing
You can create subfolders inside this directory to organize your files. However, only the files directly in this folder will be processed; files in subfolders will not be included by [`preprocess.py`](../preprocess_eeg.py).

It is a recommended practice to keep the raw data organized and well-documented for future reference or reprocessing. Blending different data sources unintentionally can lead to issues.

The preprocessed data will be saved in the [`preprocessed_data`](../preprocessed_data) folder.