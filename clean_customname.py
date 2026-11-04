import pandas as pd
import os
import sys

# Check command line argument
if len(sys.argv) != 2:
    print("Usage: python clean_customname.py [folder]")
    sys.exit(1)

folder = sys.argv[1]
output_folder = f'clean_{folder}'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# File paths
files_to_clean = [
    (f'{folder}/llama_content_topic_info.tsv', f'{output_folder}/llama_content_topic_info.tsv'),
    (f'{folder}/llama_content_document_info.tsv', f'{output_folder}/llama_content_document_info.tsv')
]

# Process each file
for input_path, output_path in files_to_clean:
    print(f"Processing {input_path}...")
    
    # Read the TSV file
    df = pd.read_table(input_path)
    
    # Clean CustomName column: keep only the first line (before \n)
    if 'CustomName' in df.columns:
        df['CustomName'] = df['CustomName'].apply(lambda x: str(x).split('\n')[0] if pd.notna(x) else x)
    
    # Save to clean folder
    df.to_csv(output_path, sep='\t', index=False)
    print(f"Saved to {output_path}")

print("\nAll files cleaned successfully!")
