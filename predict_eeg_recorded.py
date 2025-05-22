import numpy as np
import torch, os
from pylsl import StreamInlet, resolve_streams, resolve_bypred  # For real-time EEG data streaming
from trainmodel import EEGTransformer  # Import the trained EEG Transformer model
from sklearn.model_selection import train_test_split  # For splitting training data
from pynput.keyboard import Controller, Key  # For simulating keyboard button presses
import time; #use the sleep method
from preprocess_eeg import preprocess_data; # For data preprocessing
import pandas as pd;


# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

# Initialize the model
input_dim = 5  # Number of input features (e.g., EEG channels)
num_labels = 2  # Number of output classes (e.g., Left and Right)
model = EEGTransformer(input_dim=input_dim, num_labels=num_labels)

# Load pretrained model checkpoint
checkpoint_path = os.path.join("eeg_transformer_model") # CHANGE THIS PATH
checkpoint = torch.load(checkpoint_path, map_location=device)

# Load and filter the model state dictionary to match current model architecture
#model_state_dict = model.state_dict()
#filtered_state_dict = {k: v for k, v in checkpoint.items() if k in model_state_dict and model_state_dict[k].shape == v.shape}
#model_state_dict.update(filtered_state_dict)
model.load_state_dict(checkpoint)

# Prepare the model for evaluation
model.to(device)
model.eval()

# Initialize keyboard controller
keyboard = Controller()

# load recorded file
print(f"Loading data from training_data\\predict_recorded.csv...")
df = pd.read_csv("training_data\\predict_recorded.csv")  # Assumes data is stored in a CSV format

# Separate features and labels
raw_data = np.array(df.drop(columns=['timestamps']).values)  # Extract EEG channels (all columns except 'Label')

# Real-time prediction loop
buffer = []  # Buffer to store incoming EEG samples
sequence_length = 256  # Number of samples per sequence

# for handling keyboard input
lastKey=Key.left;

try:
    for i in range(raw_data.shape[0]):
        # Get a new EEG sample
        sample = raw_data[i]  # Retrieve an EEG sample from the stream
        sample = np.array(sample)  # Convert sample to numpy array

        # Smooth the sample using a moving average
        # smoothed_sample = moving_average(sample, window_size=5)

        # Normalize the sample using training data statistics
        # normalized_sample = (smoothed_sample - mean) / std
        

        # Add the normalized sample to the buffer
        buffer.append(sample)
        if len(buffer) > sequence_length:  # Maintain buffer size
            buffer.pop(0)

        # Make a prediction if the buffer has enough data
        if len(buffer) == sequence_length:
            # preprocess data
            preprocessed_buffer = preprocess_data(buffer);
            sequence = np.array(preprocessed_buffer)  # Convert buffer to numpy array
            data = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)  # Convert to PyTorch tensor

            with torch.no_grad():
                outputs = model(data)  # Forward pass through the model
                probabilities = torch.softmax(outputs, dim=1)  # Compute probabilities
                #print(f"Class Probabilities: {probabilities}")
                
                # Apply class weights
                #class_weights = torch.tensor([1.0, 1.7]).to(device)  # Example weights: higher weight for class 1
                #weighted_probabilities = probabilities * class_weights  # Scale probabilities by class weights
                prediction = torch.argmax(probabilities, dim=1).item()  # Predict based on weighted probabilities

            # Log predictions for debugging
            #print(f"Raw outputs: {outputs}")
            #print(f"Probabilities: {probabilities}")
            print(f"Predicted class: {prediction}")

            # Simulate key presses based on prediction
            if prediction == 0:  # Class 0 corresponds to "Left"
                keyboard.release(lastKey); # release the last pressed key
                lastKey=Key.left;
                keyboard.press(Key.left);
            else:  # Class 1 corresponds to "Right"
                keyboard.release(lastKey); # release the last pressed key
                lastKey=Key.right;
                keyboard.press(Key.right);

            #print(f"Predicted Action: {action}")
            time.sleep(0.003);

except KeyboardInterrupt:
    print("Prediction stopped.")

