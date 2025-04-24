from muselsl import stream, record  # Muse LSL library for streaming and recording EEG data
from datetime import datetime       # For generating timestamps
import os                           # For file and folder handling
import rmdup

# Function to start Muse streaming
def start_stream():
    """
    Starts the Muse EEG stream using muselsl.

    This function ensures that the Muse device is actively streaming data.
    If the Muse stream is already running, this function can be skipped.
    """
    try:
        stream()  # Start the Muse stream
    except Exception as e:
        print("Error starting the Muse stream:", e)

# Function to record EEG data
def record_eeg_data(duration, label):
    """
    Records EEG data using muselsl and saves it to a CSV file.

    Parameters:
        duration: int
            Duration of the recording in seconds.
        label: str
            Label to associate with the recording (e.g., "left", "right").
    
    Saves:
        A CSV file named with the label and timestamp in the current working directory.
    """
    # Get the folder where the script is located (current working directory)
    folder = os.getcwd()
    
    # Generate a unique filename with the label and current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Example: "20231124_153045"
    filename = os.path.join(folder, f"eeg_{label}_{timestamp}.csv")
    
    try:
        # Start recording EEG data
        print(f"Recording EEG data for {duration} seconds...")
        record(duration=duration, filename=filename)  # Save the recording as a CSV
        print(f"Recording saved to {filename}")
        return filename;
    except Exception as e:
        print("Error recording EEG data:", e)

# Main function
if __name__ == "__main__":
    """
    Main script to control Muse streaming and recording.

    Steps:
        1. Ensure Muse is actively streaming. If not, start the stream.
        2. Specify the duration and label for the recording.
        3. Save the recording to a CSV file in the current working directory.
    """

    # Step 1: Start Muse LSL stream (if Muse is not already streaming)
    #start_stream()  # Run this only if the Muse device is not already streaming

    # Step 2: Specify recording duration and label
    duration = 60  # Duration in seconds (Adjustable: change to desired recording length)
    label = input("Enter the label for this recording (e.g., left, right): ")

    # Step 3: Record EEG data
    filename=record_eeg_data(duration, label)
    
    # Step 4: Remove duplicate lines. will require the .exe file "remove_duplicate.exe"
    #remove_duplicate_lines() #usable in all OS systems, but slower
    os.system(".\\remove_duplicate.exe "+filename) #you can use this line if you are using windows
