import re
import sys
import csv

files = [
    "llc_run1.csv",
    "llc_run2.csv",
    "llc_run3.csv"
]

def convert_to_csv(input_file, output_file):
    """
    Convert a space-delimited file with metric data to a standard CSV file.
    
    Args:
        input_file (str): Path to the input file
        output_file (str): Path to the output file
    """
    # Read the input file
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Process the header and data rows
    data = []
    for line in lines:
        # Skip comment lines
        if line.startswith('#'):
            line = line[1:].strip()
        else:
            line = line.strip()
            
        # Skip empty lines
        if not line:
            continue
        
        # Split by variable whitespace 
        # This handles the irregular spacing in the file
        fields = re.split(r'\s+', line)
        data.append(fields)
    
    # Write the data to a CSV file
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in data:
            writer.writerow(row)
    
    print(f"Successfully converted {input_file} to {output_file}")

if __name__ == "__main__":
    for file in files:
        file_name, _ = file.split(".")
        convert_to_csv(file, file_name + "_int.csv")
