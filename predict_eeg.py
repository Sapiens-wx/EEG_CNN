import numpy as np
import torch, os
from pylsl import StreamInlet, resolve_streams, resolve_bypred  # For real-time EEG data streaming
from pynput.keyboard import Controller, Key  # For simulating keyboard button presses
import time; #use the sleep method
from preprocess_eeg import preprocess_data; # For data preprocessing
import model; # load model

def EvaluateData(data):
    """
    returns:
        prediction: (class(0 or 1), possibilities)
    """
    prediction=model.predict(processed_data);
    return (prediction>0.5?1:0,[1-prediction,prediction])

# load model
cnn=model.LoadModel("model.keras")

# Initialize keyboard controller
keyboard = Controller()

# Real-time EEG stream handling
print("Looking for an EEG stream...")
streams = resolve_bypred("type='EEG'")  # Resolve EEG stream
inlet = StreamInlet(streams[0])  # Connect to the first available EEG stream
print("EEG stream found. Starting real-time classification.")

# Define a smoothing function for incoming EEG samples
def moving_average(data, window_size=5):
    """
    Applies a moving average to smooth incoming EEG data.

    Parameters:
        data: 1D numpy array of EEG signal data.
        window_size: int, the number of samples to average over.

    Returns:
        Smoothed EEG data as a 1D numpy array.
    """
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Real-time prediction loop
buffer = []  # Buffer to store incoming EEG samples
sequence_length = 256  # Number of samples per sequence

# used for handle keyboard input
lastKey=Key.left;

# predict interval in seconds
predictInterval=0.5;
programStartTime=time.time();

try:
    while True:
        # record start time
        startTime=time.time();
        print(f"time={startTime-programStartTime}");
        # Get a new EEG sample
        samples, _ = inlet.pull_chunk()  # Retrieve EEG samples from the stream
        if samples:
            for sample in samples:
                buffer.append(np.array(sample));
                if len(buffer)>sequence_length:
                    buffer.pop(0);

        # Make a prediction if the buffer has enough data
        if len(buffer) == sequence_length:
            # preprocess data
            preprocessed_buffer = preprocess_data(buffer);
            sequence = np.array(preprocessed_buffer)  # Convert buffer to numpy array

            prediction = EvaluateData(sequence)  # Forward pass through the model
            probabilities = prediction[1]  # Compute probabilities
            predict_class=prediction[0] # class (left or right)
                
            # Log predictions for debugging
            #print(f"Raw outputs: {outputs}")
            #print(f"Probabilities: {probabilities}")
            print(f"Predicted class: {predict_class}")
            print(f"Class Probabilities: {probabilities}")

            # Simulate key presses based on prediction
            if predict_class == 0:  # Class 0 corresponds to "Left"
                keyboard.release(lastKey); # release the last pressed key
                lastKey=Key.left;
                keyboard.press(Key.left);
            else:  # Class 1 corresponds to "Right"
                keyboard.release(lastKey); # release the last pressed key
                lastKey=Key.right;
                keyboard.press(Key.right);

            #print(f"Predicted Action: {action}")
        timeElapsed=time.time()-startTime;
        if predictInterval-timeElapsed>0:
            time.sleep(predictInterval-timeElapsed);

except KeyboardInterrupt:
    print("Prediction stopped.")

