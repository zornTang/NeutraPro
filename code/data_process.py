import argparse
import pandas as pd

def convert_pkl_to_csv_and_pkl(pkl_file_path, csv_file_path, balanced_pkl_file_path):
    """
    Convert a .pkl file to a balanced .csv file and save the balanced data as a .pkl file.

    Args:
        pkl_file_path (str): The path to the input .pkl file.
        csv_file_path (str): The path to the output .csv file.
        balanced_pkl_file_path (str): The path to save the balanced .pkl file.

    Returns:
        None
    """
    try:
        # Load the .pkl file
        data = pd.read_pickle(pkl_file_path)
        
        # Print the data for inspection
        print("Original data preview:")
        print(data.head())  # Print the first 5 rows of the data
        
        # Filter rows where 'human' column is 1
        human_1_data = data[data['human'] == 1]
        
        # Randomly sample the same number of rows where 'human' column is 0
        human_0_data = data[data['human'] == 0].sample(n=len(human_1_data), random_state=42)
        
        # Combine the two datasets
        balanced_data = pd.concat([human_1_data, human_0_data])
        
        # Shuffle the combined dataset
        balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Print the balanced data for inspection
        print("Balanced data preview:")
        print(balanced_data.head())
        
        # Save the balanced data to a .csv file
        balanced_data.to_csv(csv_file_path, index=False)
        print(f"Successfully saved balanced data to {csv_file_path}")
        
        # Save the balanced data to a .pkl file
        balanced_data.to_pickle(balanced_pkl_file_path)
        print(f"Successfully saved balanced data to {balanced_pkl_file_path}")
    except Exception as e:
        # Handle exceptions and print error messages
        print(f"Conversion failed: {e}")

def convert_pkl_to_fasta(pkl_file_path, fasta_file_path):
    """
    Convert a .pkl file to a FASTA file using the 'index' column as the header.

    Args:
        pkl_file_path (str): The path to the input .pkl file.
        fasta_file_path (str): The path to the output .fasta file.

    Returns:
        None
    """
    try:
        # Load the .pkl file
        data = pd.read_pickle(pkl_file_path)
        
        # Ensure the 'index' column exists
        if 'index' not in data.columns:
            raise ValueError("The .pkl file does not contain an 'index' column.")
        
        # Ensure the 'sequence' column exists
        if 'sequence' not in data.columns:
            raise ValueError("The .pkl file does not contain a 'sequence' column.")
        
        # Open the FASTA file for writing
        with open(fasta_file_path, 'w') as fasta_file:
            headers = []  # List to store headers for inspection
            for _, row in data.iterrows():
                # Use the 'index' column as the header and 'sequence' column as the sequence
                header = f">{row['index']}"  # Use the value in the 'index' column
                sequence = row['sequence']  # Use the value in the 'sequence' column
                fasta_file.write(f"{header}\n{sequence}\n")
                headers.append(header)  # Collect headers
        
        # Print all headers for inspection
        print("FASTA headers:")
        for header in headers:
            print(header)
        
        print(f"Successfully saved FASTA file to {fasta_file_path}")
    except Exception as e:
        # Handle exceptions and print error messages
        print(f"FASTA conversion failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Process .pkl files and convert to .csv, .pkl, and .fasta formats.")
    parser.add_argument("--pkl_file", required=True, help="Path to the input .pkl file.")
    parser.add_argument("--csv_file", required=True, help="Path to the output .csv file.")
    parser.add_argument("--balanced_pkl_file", required=True, help="Path to the output balanced .pkl file.")
    parser.add_argument("--fasta_file", required=True, help="Path to the output .fasta file.")
    args = parser.parse_args()

    # Convert .pkl to balanced .csv and .pkl
    convert_pkl_to_csv_and_pkl(args.pkl_file, args.csv_file, args.balanced_pkl_file)

    # Convert the balanced .pkl to FASTA
    convert_pkl_to_fasta(args.balanced_pkl_file, args.fasta_file)

if __name__ == "__main__":
    main()