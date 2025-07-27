# Table of Contents
1. [EEG Classification System Usage](#eeg-classification-system---versatile-version)
2. [EGQ](#eeg-game-questionnaire-egq)
3. [GEQ](#game-experience-questionnaire-geq)
4. [SUS](#system-usability-scale-sus)
5. [Instructions](#Instructions)

# EEG Classification System - Versatile Version

A comprehensive EEG (Electroencephalogram) signal classification system that supports real-time data acquisition, preprocessing, model training, and real-time classification. The system can recognize different brainwave patterns (such as left, right, neutral, etc.) and convert classification results into keyboard inputs through Socket communication.

## Features

- **Real-time EEG Data Acquisition**: Supports Muse devices for data collection via LSL protocol
- **Multiple Deep Learning Models**: Supports four model architectures: CNN, Transformer, CNN+LSTM, DaViT
- **Data Preprocessing**: Automatic filtering and segmentation of EEG signals
- **Real-time Classification**: Real-time EEG classification through Socket communication
- **Keyboard Mapping**: Converts classification results into keyboard key inputs
- **Flexible Label System**: Supports various EEG state labels (left, right, neutral, transition states, etc.)

## File Structure

```
VersatileVersion/
├── README.md                    # Project documentation
├── labels.py                    # Label definition and validation
├── models.py                    # Deep learning model definitions
├── record_eeg.py               # EEG data acquisition
├── preprocess_eeg.py           # Data preprocessing
├── train_model.py              # Model training
├── classify_eeg_socket.py      # Real-time classification service
├── socket_to_keyboard.py       # Socket to keyboard mapping or engine
├── recorded_data/              # Raw EEG data storage
├── preprocessed_data/          # Preprocessed data
└── models/                     # Trained model files
```

## Dependencies

Make sure to install the following Python packages:

**Strongly recommended to use Python 3.12 version.**

```bash
pip install tensorflow
pip install numpy pandas
pip install scipy scikit-learn
pip install pylsl muselsl
pip install tqdm
pip install keyboard
```

## Usage Workflow

### 1. Data Collection

First, collect EEG data for training:

```bash
python record_eeg.py --user name
python record_eeg.py -cues 'l,r,n' --user name
python record_eeg.py -cues 'l,r,n' -loop 4 --user name
```

Parameter description:
- `-cues`: [deafult='l,r,n'] an array of cues you want to generate. DO NOT repeat the cues. For example, '-cues "l,r,n"'
- `-loop`: [default=4] how many times do you want to repeat the array of cues
- `--user`: will create a subfolder under recorded-data folder named "name", designed for users to seperate each individual test round for different people or for different set of training settings and able to name it.
- config.py: recordEEG.taskLength: number of seconds for each task
- config.py: recordEEG.transitionLength: number of seconds for transition between each task (preprocess_eeg.py will ignore the eeg data duration this transition stage)

### 2. Data Preprocessing

Preprocess the collected raw data:

```bash
python preprocess_eeg.py --user name -labels "left,right,neutral" -windowSize 256 -slidingWindow 128
```

Parameter description:
- `--user`: the subfolder of recorded_data you want to preprocess of, and then store it under the subfolder created in preprocessed_data named "name"
- `-labels`: Comma-separated label list. must be in the same order as you entered in record_eeg.py
- `-windowSize`: Time window size (number of samples)
- `-slidingWindow`: Sliding window step size
- `-asCSV`: [optional] [an integer] if > 0, then save at most [asCSV] segments independently as .csv files
- `-doBandPass`: [optional] [0 or 1] if ==1, then apply bandpass filter
- `-normalizationMethod`: [optional] [none, z-score, min-max, robust]. default as z-score. The normalization method used to normalize the signal.
- `-featureExtraction`: [optional] [none, fft, sfft, wavelet, wave] default as none. what feature extraction method do we want?
  - use "fft" for fast fourier transform (note that if you use fft as feature extraction method, you'll have to add "-model cnn_featureExtraction" while using train_model.py, instead of "-model cnn")
  - use "wave" to extract beta and theta waves (note that if you use wave as feature extraction method, you'll have to add "-model cnn_featureExtraction" while using train_model.py, instead of "-model cnn")

### 3. Model Training

Train the classification model:

```bash
python train_model.py --user name -model CNN -labels "left,right,neutral" -epochs 100 -windowSize 256 -slidingWindow 128 0.8
```

Alternatively, if you have a preprocessed file, you can skip the label recognition step:

```bash
python train_model.py --user name -model CNN -preprocessedFilePath "path/to/preprocessed_file.npy" -epochs 100
python train_model.py --user name -model CNN -label "left,right,neutral" -epochs 100
```

Parameter description:
- `--user`: declare which user or which data set you are using and create a subfolder under models named "name" to store models that were trained for specific users
- `-model`: Model type (CNN, Transformer, CNN+LSTM, DaViT, CNN_featureExtraction)
  - CNN_featureExtraction: the different between CNN_featureExtraction and CNN is their kernel size in conv1D layers. CNN_featureExtraction has smaller kernel size to fit the small input size (preprocessed with feature extraction method of "fft" or "wave")
- `-labels`: Label list
- `-epochs`: Number of training epochs
- `-windowSize`: Time window size
- `-slidingWindow`: Sliding window step size (number of samples). When segmenting EEG data, each window advances by this step size, determining the overlap between windows. For example, with `-windowSize 256 -slidingWindow 128`, each window contains 256 samples and moves forward by 128 samples, resulting in 50% overlap. Smaller step sizes produce more samples, potentially improving model performance but increasing data volume and computation.
- `-trainDataRatio`: Last parameter is the training data ratio (e.g., `0.8` means 80% of data is used for training and 20% for testing).
- `-preprocessedFilePath`: Path to a preprocessed `.npy` file. If provided, skips the label recognition logic and directly uses the specified file.

### 4. Real-time Classification

Start the real-time classification service:
```bash
python classify_eeg_socket.py -model CNN -labels "left,right,neutral" -windowSize 256 -slidingWindow 128 -host 127.0.0.1 -port 9000
```

### 5. Keyboard Mapping

Convert classification results to keyboard input:

```bash
python socket_to_keyboard.py -host 127.0.0.1 -port 9000 -threshold 0.7 -runClassificationToSocket -window 256 -slidingwindow 128
```

Parameter description:
- `-threshold`: Classification confidence threshold
- `-runClassificationToSocket`: Automatically start classification service
- `-windowSize`: Window size (number of samples)
- `-slidingWindow`: Sliding window step size (number of samples). When segmenting EEG data, each window advances by this step size, determining the overlap between windows. For example, with `-windowSize 256 -slidingWindow 128`, each window contains 256 samples and moves forward by 128 samples, resulting in 50% overlap. Smaller step sizes produce more samples, potentially improving model performance but increasing data volume and computation.

## Supported Label Types

The system supports the following EEG state labels:

- **Basic States**:
  - `left/l/L`: Left motor imagery
  - `right/r/R`: Right motor imagery
  - `neutral/n/N`: Neutral/rest state

- **Transition States**:
  - `left-to-right/l2r/L2R`: Left to right transition
  - `left-to-neutral/l2n/L2N`: Left to neutral transition
  - `right-to-left/r2l/R2L`: Right to left transition
  - `right-to-neutral/r2n/R2N`: Right to neutral transition
  - `neutral-to-left/n2l/N2L`: Neutral to left transition
  - `neutral-to-right/n2r/N2R`: Neutral to right transition

- **Future Works**:
  - If you plan to add more labels, update the `label_map` dictionary in [`labels.py`](./labels.py#L2) accordingly. All labels should be defined in the `label_map` for proper validation and processing. No other work is needed for the system to recognize new labels, as long as they are added to the `label_map`.

## Supported Model Architectures

### 1. CNN
Basic Convolutional Neural Network, suitable for processing time-domain EEG signals.

### 2. Transformer
Attention-based Transformer model that can capture long-range dependencies.

### 3. CNN+LSTM
Combines convolutional and Long Short-Term Memory networks, suitable for temporal feature extraction.

### 4. DaViT (Dual Attention Transformer)
Dual Attention Transformer providing stronger feature representation capabilities.

### Future Works
  - If you plan to add more model architectures, implement them in [`models.py`](./models.py) as a new function. Add name and aliases as needed in [`MODEL_ALIASES`](./models.py#L1). Then add the recognition logic to the [`LoadModel`](./models.py#L19).

## Configuration

### Window Parameters
- `windowSize`: Time window size for each classification sample (number of samples)
- `slidingWindow`: Sliding window step size, controls sample overlap

### Socket Communication
- Default uses localhost:9000 for communication
- Classification results transmitted in JSON format
- Supports confidence threshold filtering

## Important Notes

1. **Device Connection**: Ensure your EEG device is properly connected and transmitting data through an LSL stream. The system is compatible with any device that outputs EEG data via LSL; while `muselsl` and 'BlueMuse' are commonly used for Muse devices, any LSL-compliant stream will work.
2. **Participant Comfort**: Avoid causing frustration or fatigue for participants during data collection. Ensure breaks are provided and instructions are clear to maintain motivation and data quality.

   **Display Placement**: If possible, ensure the game screen is within the participant's 30° field of view. This may help reduce potential motion sickness. The recommended minimum viewing distance can be calculated using the formula:

   ```
   d = w / (2 * tan(15°))
   ```

   Where:
   - `d`: Minimum viewing distance (in the same units as `w`)
   - `w`: Width of the game screen
   - `tan(15°) ≈ 0.2679`

   Example minimum distances for full-screen operation:
   - For a 24-inch display (16:9 aspect ratio, width ≈ 20.9 inches): `d ≈ 39.1` inches (≈ 99.3 cm)
   - For a 27-inch display (16:9 aspect ratio, width ≈ 23.5 inches): `d ≈ 44.0` inches (≈ 111.8 cm)

   These are only recommendations and may vary based on individual preferences and experimental setups.
3. **Model Selection**: Choose appropriate model architecture based on data complexity. However, the system is designed to be flexible and can adapt to different model architectures as needed. You can keep the recorded data, or preprocessed .npy file under ['preprocessed_data'](./preprocessed_data/), and use it for training with different models without needing to re-record data.
4. **Parameter Tuning**: Adjust window size and threshold parameters according to specific application scenarios. Further tuning is not provided by commandline arguments, but you can modify the parameters in the respective Python files directly.

## Troubleshooting

### Common Issues

1. **EEG Stream Not Found**
   - Check Muse device connection
   - Verify LSL stream is working properly

2. **Model Training Failed**
   - Check if preprocessed data exists
   - Verify label format is correct
   - DON‘T RENAME PREPROCESSED DATA FILES AFTER THEY ARE CREATED. THIS WILL CAUSE UNEXPECTED ISSUES IN TRAINING AND CLASSIFICATION.

3. **Socket Connection Failed**
   - Check if port is occupied
   - Verify firewall settings

4. **Poor Classification Accuracy**
   - Increase training data volume
   - Adjust model architecture or parameters
   - Improve data collection quality
   - Perhaps try different preprocessing methods
   - Make sure there's no overfitting issue

5. **Label Issues**
   - Ensure all labels are correctly defined in the `label_map`
   - Verify label format is consistent across all data samples. This is already handled in the `labels.py` file. Don't rename files or labels after they are defined, as it may cause issues with data processing and model training, since label recognition is based on file names.


## Extension Development

The system is designed with modular architecture for easy extension:

- Add new model architectures to `models.py`
- Extend label system in `labels.py`
- Customize preprocessing methods in `preprocess_eeg.py`
- Add new output methods (such as OSC, MIDI, etc.)

## License

[Add license information here]

## Contributing

[Add contribution guidelines here]

---

**Note**: This system is intended only for research and development purposes. It must never be used for clinical diagnosis, medical decision-making, or any other unapproved medical applications. Use of this system for any medical or healthcare purposes without proper regulatory review and approval is strictly prohibited.

**Ethics Notice**: Any research based on this system must ensure the protection of participant privacy and address other ethical considerations.

# EEG Game Questionnaire (EGQ)
This is a five-question questionnaire designed by us to target specifically at evaluating EEG-controlled games.
||Question|Strongly Disagree<br>(1)|Disagree<br>(2)|Neutral<br>(3)|Agree<br>(4)|Strongly Agree<br>(5)|
|-|-|-|-|-|-|-|
|1|I think using an EEG controller makes the game more fun than a traditional controller||||||
|2|The EEG has high mental demands that I cannot make timely decisions in the game||||||
|3|I prefer playing this EEG game over participating in a traditional EEG data collection session||||||
|4|I felt mentally exhausted while playing this game using EEG control||||||
|5|I felt the EEG controller responded to my intention||||||

# Game Experience Questionnaire (GEQ)
[Download](https://pure.tue.nl/ws/files/21666907/Game_Experience_Questionnaire_English.pdf)

Use the core module of the questionnaire is enough. Other parts are for multiplayer and voluntary playing

# System Usability Scale (SUS)

[Download](https://digital.ahrq.gov/sites/default/files/docs/survey/systemusabilityscale%2528sus%2529_comp%255B1%255D.pdf)

You need the **10 responses** from a participant. Each response is a score from **1 to 5**.

## 🔢 Step 1: Adjust Each Item Score

- **Odd-numbered items** (1, 3, 5, 7, 9):  
  `Adjusted = Score - 1`

- **Even-numbered items** (2, 4, 6, 8, 10):  
  `Adjusted = 5 - Score`

## 📊 Step 2: Sum All Adjusted Scores

Add the 10 adjusted values. The total will be between **0 and 40**.

## ✖️ Step 3: Multiply by 2.5

```text
SUS = (Sum of Adjusted Scores) × 2.5
```

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
```python train.py```

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

## Instructions

### Command Line Usage Guide for All Scripts

#### 1. record_eeg.py
Record EEG data for training. Supports 9 labels.

**Arguments:**
- `-lable <label>`: Specify the label (choose from: left, right, neutral, left to right, right to left, left to neutral, right to neutral, neutral to left, neutral to right). If not provided, you will be prompted.
- `-t <duration>`: Recording duration in seconds. If not provided, you will be prompted.
- `-enableProgress`: Show a progress bar during recording.

**Example:**
```
python record_eeg.py -lable left -t 10 -enableProgress
```

---

#### 2. preprocess_eeg.py
Preprocess all EEG .csv files in `training_data/`. No arguments needed.

**Example:**
```
python preprocess_eeg.py
```

---

#### 3. train.py
Train the model using preprocessed data. No arguments needed.

**Example:**
```
python train.py
```

---

#### 4. evaluate_training_data.py
Evaluate model performance on training data.

**Arguments:**
- `-j <int>`: Number of parallel processes (default: 1)
- `-testCount <int>`: Number of test repetitions per sample (default: 5)
- `-windowTime <int>`: Window size in milliseconds (default: 500)
- `-categories <label1> <label2> ...`: Only evaluate specified categories (e.g., left right neutral)

**Example:**
```
python evaluate_training_data.py -j 4 -testCount 10 -windowTime 500 -categories left right neutral
```

---

#### 5. predict_eeg.py
Real-time EEG prediction. No arguments needed.

**Example:**
```
python predict_eeg.py
```

---

#### 6. predict_eeg_judge_with_keyboard.py
Real-time prediction with manual labeling or transition test.

**Arguments:**
- `--transitionTest`: Enable transition test mode (see instructions printed at startup). If not provided, runs in manual labeling mode.

**Example:**
```
python predict_eeg_judge_with_keyboard.py --transitionTest
```

---

#### 7. train_switch_model.py
Train a model with selectable architecture (CNN, Transformer, or CNN+LSTM).

**Arguments:**
- `-model <type>`: Specify the model type. Options: `CNN`, `Transformer`, `CNN+LSTM` (case-insensitive, spaces allowed, e.g. `-model "CNN+LSTM"`).
- `--epoch`, `-epoch`, `-epochs <int>`: Number of training epochs (default: 100).
- `-printResult`: Print model summary after training.

**Example:**
```
python train_switch_model.py -model CNN -epochs 200 -printResult
python train_switch_model.py -model "CNN+LSTM" -epoch 150
python train_switch_model.py -model transformer
```

---

#### 8. General Notes
- All scripts must be run from the project root or the correct subfolder as described above.
- For label arguments, always use one of the 9 supported labels exactly as listed.
- For more details, see the comments at the top of each script.

