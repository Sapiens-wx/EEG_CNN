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
python record_eeg.py -label left -duration 60 -progressbar
python record_eeg.py -label left-to-right -duration 60 -progressbar
python record_eeg.py -label l2n -duration 60 -progressbar
```

Parameter description:
- `-label`: Data label (supports: left/l, right/r, neutral/n, left-to-right/l2r/ltr etc.)
- `-duration`: Collection duration (seconds)
- `-progressbar`: Show progress bar

### 2. Data Preprocessing

Preprocess the collected raw data:

```bash
python preprocess_eeg.py -labels "left,right,neutral" -windowSize 256 -slidingWindow 128
```

Parameter description:
- `-labels`: Comma-separated label list
- `-windowSize`: Time window size (number of samples)
- `-slidingWindow`: Sliding window step size

### 3. Model Training

Train the classification model:

```bash
python train_model.py -model CNN -labels "left,right,neutral" -epochs 100 -windowSize 256 -slidingWindow 128 0.8
```

Alternatively, if you have a preprocessed file, you can skip the label recognition step:

```bash
python train_model.py -model CNN -preprocessedFilePath "path/to/preprocessed_file.npy" -epochs 100
python train_model.py -model CNN -label "left,right,neutral" -epochs 100
```

Parameter description:
- `-model`: Model type (CNN, Transformer, CNN+LSTM, DaViT)
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