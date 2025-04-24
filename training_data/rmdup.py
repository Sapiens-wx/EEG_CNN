import os
import pandas as pd
import csv

def process_csv_file2(file):
    """
    Reads a CSV file and creates a new file (file.tmp) containing:
    - The first line (header) unchanged
    - Subsequent lines only if their first column value increases
    """
    try:
        with open(file, 'r') as infile, open(file + '.tmp', 'w', newline='') as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            
            # Copy header/first line
            header = next(reader)
            writer.writerow(header)
            
            # Initialize previous value with first data row
            try:
                prev_row = next(reader)
                prev_value = float(prev_row[0])
                writer.writerow(prev_row)
            except StopIteration:
                return  # Empty file after header
            
            # Process remaining rows
            total=0;
            remove_cnt=0;
            for row in reader:
                try:
                    current_value = float(row[0])
                    if current_value > prev_value:
                        writer.writerow(row)
                        prev_value = current_value
                    else:
                        remove_cnt+=1;
                    total+=1;
                except (ValueError, IndexError):
                    continue  # Skip rows with invalid/missing first column
            print(total)
            print(f"{file}: {remove_cnt}/{total} lines removed")   
            infile.close()
        os.remove(file)  # 1: remove infile
        os.rename(file + '.tmp', file)  # 2: rename outfile to infile
    except FileNotFoundError:
        print(f"Error: File '{file}' not found")
    except Exception as e:
        print(f"Error processing file: {e}")
def process_csv_file(file):
    outfile_name=file+".tmp";
    outfile=open(outfile_name, "w");
    try:
        # Read CSV, treating all columns as strings initially
        df = pd.read_csv(file, dtype=str, keep_default_na=False)
        
        lastTime=0;
        removed_cnt=0;
        total=0;
        # iterate files
        for idx, row in df.iterrows():
            print(f"line {idx},({idx==1}):{",".join(str(x) for x in row)}")
            if len(row) == 0:  # Skip empty rows
                continue
            if idx==1:
                outfile.write(",".join(str(x) for x in row))
                continue
            first_entry = str(row.iloc[0]).strip()
            try:
                timeStamp = float(first_entry)
                if(timeStamp>lastTime):
                    outfile.write(",".join(str(x) for x in row))
                    lastTime=timeStamp;
                else:
                    ++removed_cnt;
                ++total;
            except ValueError:
                print(f"Row {idx+1}: Could not convert '{first_entry}' to decimal")
        print(f"{file}: {removed_cnt}/{total} lines removed")
    except Exception as e:
        print(f"Error processing {file}: {str(e)}")
    outfile.close()

def remove_dup_lines():
    csv_files = [f for f in os.listdir() if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found")
        return

    for filename in csv_files:
        process_csv_file2(filename)
        

if __name__ == "__main__":
    remove_dup_lines()