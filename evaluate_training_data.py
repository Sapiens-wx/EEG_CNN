import numpy as np
import os
import pandas as pd
from preprocess_eeg import preprocess_data
import tensorflow as tf

# Load the model
try:
    model = tf.keras.models.load_model('model.keras')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

def predict_file(file_path, n_predictions=10):
    # Load and preprocess the file
    df = pd.read_csv(file_path)
    data = df.drop(columns=['timestamps']).values
    
    # Process in windows of 256 samples
    sequence_length = 256
    all_window_predictions = []
    
    for i in range(0, len(data) - sequence_length + 1, sequence_length):
        window = data[i:i + sequence_length]
        if len(window) == sequence_length:
            window_predictions = []
            # Preprocess the window
            preprocessed_window = preprocess_data(window)
            sequence = np.array(preprocessed_window)
            
            # Make n predictions for this window
            sequence = np.expand_dims(sequence, axis=0)
            for _ in range(n_predictions):
                outputs = model.predict(sequence, verbose=0)
                prediction = np.argmax(outputs[0])
                window_predictions.append(prediction)
            
            # Get most common prediction for this window
            most_common = max(set(window_predictions), key=window_predictions.count)
            all_window_predictions.append(most_common)
            
            # Print detailed prediction results for this window
            left_count = window_predictions.count(0)
            right_count = window_predictions.count(1)
            rest_count = window_predictions.count(2)
            print(f"  Window {i//sequence_length + 1}: Left={left_count}, Right={right_count}, Rest={rest_count} -> {most_common}")
    
    # Return most common prediction across all windows
    if all_window_predictions:
        final_prediction = max(set(all_window_predictions), key=all_window_predictions.count)
        
        # Calculate overall statistics
        total_left = sum(1 for p in all_window_predictions if p == 0)
        total_right = sum(1 for p in all_window_predictions if p == 1)
        total_rest = sum(1 for p in all_window_predictions if p == 2)
        total_windows = len(all_window_predictions)
        
        print(f"  Overall: Left={total_left}/{total_windows}, Right={total_right}/{total_windows}, Rest={total_rest}/{total_windows}")
        return final_prediction
    return -1

# Process all CSV files in the training_data directory
training_data_dir = "training_data"
results = []

for file in os.listdir(training_data_dir):
    if file.endswith(".csv"):
        print(f"\nProcessing {file}:")
        file_path = os.path.join(training_data_dir, file)
        prediction = predict_file(file_path)
        class_name = "left" if prediction == 0 else "right" if prediction == 1 else "rest"
        print(f"Final prediction: {class_name} ({prediction})")
        results.append((file, class_name, prediction))

# Print summary of correct/incorrect predictions
print("\nSummary of predictions:")
correct = 0
total = 0
for file, predicted_class, prediction in results:
    true_class = None
    if "left" in file.lower():
        true_class = "left"
    elif "right" in file.lower():
        true_class = "right"
    elif "rest" in file.lower():
        true_class = "rest"
    
    if true_class is not None:
        is_correct = true_class == predicted_class
        mark = "✓" if is_correct else "✗"
        if is_correct:
            correct += 1
        total += 1
        print(f"{mark} {file}: True={true_class}, Predicted={predicted_class}")

if total > 0:
    print(f"\nAccuracy: {correct}/{total} = {correct/total*100:.2f}%")
else:
    print("\nNo files were evaluated.")
