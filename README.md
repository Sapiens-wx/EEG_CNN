# eeg_racing_game
Note: Only compatiable with Windows OS currently

## Table of Contents
1. [Install TuxRacer](#step-1-install-tuxracer)
2. [Testing the Muse Meditation App](#step-2-testing-the-muse-meditation-app)
3. [Install BlueMuse](#step-3-install-bluemuse)
4. [Setup Python](#step-4-setup-python)
5. [Record EEG Data](#step-5-record-eeg-data)
6. [Combine EEG Data](#step-6-combine-eeg-data)
7. [Preprocess EEG Data](#step-7-preprocess-eeg-data)
8. [Train the Model](#step-8-train-the-model)
9. [Run the Real-Time Controller](#step-9-run-the-real-time-controller)


## Step 1: Install TuxRacer
Tux Racer is a free, open-source game where you control a penguin down a slope. Here’s how to install it:

Download Tux Racer on SourceForge from their [website.](https://tuxracer.sourceforge.net/download.html#Windows)\
Run the installer and follow the prompts.

## Step 2: Testing the Muse Meditation App

Before connecting the Muse headset to your computer, test it using the official Muse Meditation App to ensure it is functioning correctly.

Download the Muse Meditation App from the App Store or Google Play.\
Pair your Muse device with your smartphone via Bluetooth.\
Open the app and follow the guided instructions to test EEG signal acquisition.\
Ensure the Muse device is sitting on your head and the app shows stable connectivity.


## Step 3: Install BlueMuse

BlueMuse is software that allows the Muse 2 headband to communicate over Bluetooth.

Download BlueMuse from the official GitHub [repo.](https://github.com/kowalej/BlueMuse)\
Run the installer and follow the prompts to complete the installation.\
Ensure your Muse 2 headband is fully charged and in pairing mode.\
Open BlueMuse and connect to the Muse headband.

## Step 4: Setup Python 

Ensure the following python libraries are installed:\
```torch```\
```numpy```\
```pandas```\
```scikit-learn```\
```pynput```\
```pylsl```

## Step 5: Record EEG Data

Use the record_eeg.py script to collect EEG data for training the model.

Place the Muse device on your head and ensure it’s streaming via BlueMuse.\
Run the script to start recording (Make sure you cd into the training_data folder):\
```python record_eeg.py```

The script will save the recorded EEG data as a .csv file in the current directory.\
Repeat this process for different labels (e.g., left, right) to collect sufficient data for training.\
Minimum recommended abount is about 6 times per direction (The more the better!!!)

## Step 6: Combine EEG Data

Combine all the recorded .csv files into a single dataset using the combinedata.py script.

Place all .csv files into a folder.\
Update the data_folder variable to match your folder path\
Run the script:\
```python combinedata.py```

The script will create a combined dataset named combined_training_data.csv in the current directory.

## Step 7: Preprocess EEG Data

Preprocess the combined dataset using the preprocess_eeg.py script.

Ensure the combined_training_data.csv file is in the correct directory.\
Update the input_file and save_folder variables in preprocess_eeg.py\
Run the script:\
```python preprocess_eeg.py```

The script will:\
Filter the data using a bandpass filter (may need to adjust adjust lowcut, highcut, and fs in the future).\
Normalize and segment the data.\
Save processed files as eeg_segments.npy and eeg_labels.npy

## Step 8: Train the Model

Train the EEG Transformer model using the train_eeg_transformer.py script.

Ensure the .npy files (eeg_segments.npy and eeg_labels.npy) are in the specified folder.\
Update the paths in the script if necessary\
Run the script:\
```python trainmodel.py```

The script will:\
Train the model for 5 epochs (adjust epochs in the script if needed, the more the better!!).\
Save the trained model as eeg_transformer_model.pth.

## Step 9: Run the Real-Time Controller

Control Tux Racer in real-time using the eeg_racing_game.py script.

Connect your Muse headset and ensure it is streaming via BlueMuse.\
Launch Tux Racer.\
Update the paths in eeg_racing_game.py\
```python predict_eeg.py```

The script will:\
Read real-time EEG data and normalize it.\
Predict actions ("Left" or "Right") based on EEG signals.\
Simulate arrow key presses to control Tux Racer.

