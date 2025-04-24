import pandas as pd
import os

# Path to the folder containing the training data
data_folder = os.path.join("training_data" ) # CHANGE THIS PATH AS NEEDED

# Initialize an empty list to store dataframes
dataframes = []

# Loop through each file in the folder
for file in os.listdir(data_folder):
    if file.endswith(".csv"):  # Process only CSV files
        # Load the CSV file into a dataframe
        df = pd.read_csv(os.path.join(data_folder, file))
        
        # Extract label from the filename and assign it to a new column
        if "left" in file.lower():
            df['Label'] = 0  # Assign label 0 for "left"
        elif "right" in file.lower():
            df['Label'] = 1  # Assign label 1 for "right"
        else:
            print(f"Warning: No label assigned for file {file}. Skipping...")
            continue  # Skip files without "left" or "right" in the filename
        
        # Append the dataframe to the list
        dataframes.append(df)

# Combine all dataframes into one
if dataframes:
    combined_df = pd.concat(dataframes, ignore_index=True)  # Combine all loaded dataframes
    
    # Save the combined dataframe to a CSV file
    output_file = "combined_training_data.csv"  # CHANGE OUTPUT FILE NAME OR PATH IF NEEDED
    combined_df.to_csv(output_file, index=False)
    print(f"Combined data saved to {output_file}")
else:
    print("No valid CSV files found or processed. Exiting...")
