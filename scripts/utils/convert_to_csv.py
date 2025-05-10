import argparse
import re
import csv


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
        
        # Skip lines indicating Timestamps
        if line.startswith("Timestamp"):
            continue

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Filename of the txt file to convert")
    parser.add_argument("--out", type=str, help="The filename of the output file")
    args = parser.parse_args()

    file_name = args.file.split("/")[-1].split(".")[0].strip()
    convert_to_csv(args.file, f"{args.out}/{file_name}.csv")
