import numpy as np
import os
from pylsl import StreamInlet, resolve_streams, resolve_bypred  # For real-time EEG data streaming
from pynput.keyboard import Controller, Key  # For simulating keyboard button presses
import time; #use the sleep method
from preprocess_eeg import preprocess_data; # For data preprocessing
import model; # load model
import communication; # communication with Unity game
import config; # get ip_address and port

def EvaluateData(data):
    """
    returns:
        prediction: (class(0 or 1), possibilities)
    """
    data=data.reshape((1,data.shape[0],data.shape[1]))
    prediction=cnn.predict(data, verbose=0); # verbose=0 disables output
    if prediction[0]>0.5:
        return (1,[1-prediction[0],prediction[0]])
    return (0,[1-prediction[0],prediction[0]])

# callback for receiving label from the game. gets one byte: the user is thinking of left/right/neither 0/1/2
def OnReceiveLabelFromGame(data):
    curLabel=int.from_bytes(data[0],'big');

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

# is the user currently thinking left/right? left: 0, right: 1, neither(rest state): 2
# used for the meteorite game
curLabel=2;
communication.connect_async(config.ip_address, config.port, OnReceiveLabelFromGame);

try:
    while True:
        # record start time
        startTime=time.time();
        print(f"time={startTime-programStartTime}, curLabel={curLabel}");
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

