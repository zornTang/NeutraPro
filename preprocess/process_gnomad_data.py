#!/usr/bin/env python3
"""
Script to process gnomAD v4.1 constraint metrics data.
Extracts data, removes duplicates, filters out missing LOEUF values,
keeps only protein_coding transcripts, and retrieves protein sequences from Ensembl BioMart.
"""

import pandas as pd
import numpy as np
from biomart import BiomartServer
from typing import List, Dict
import time

def get_protein_sequences_biomart(transcript_ids: List[str], batch_size: int = 100) -> Dict[str, str]:
    """
    Get protein sequences for transcript IDs using biomart library.
    
    Args:
        transcript_ids: List of Ensembl transcript IDs
        batch_size: Number of transcripts to query per batch
        
    Returns:
        Dictionary mapping transcript ID to protein sequence
    """
    print("Connecting to Ensembl BioMart...")
    
    try:
        # Connect to Ensembl
        server = BiomartServer("http://www.ensembl.org/biomart")
        dataset = server.datasets['hsapiens_gene_ensembl']
        
        sequences = {}
        
        # Process in batches
        for i in range(0, len(transcript_ids), batch_size):
            batch = transcript_ids[i:i + batch_size]
            print(f"Querying batch {i//batch_size + 1}/{(len(transcript_ids) + batch_size - 1)//batch_size} ({len(batch)} transcripts)...")
            
            try:
                # Query for protein sequences
                response = dataset.search({
                    'filters': {
                        'ensembl_transcript_id': batch
                    },
                    'attributes': [
                        'ensembl_transcript_id',
                        'peptide'
                    ]
                })
                
                # Parse response
                for line in response.iter_lines(decode_unicode=True):
                    if line.strip():
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            # BioMart returns: sequence \t transcript_id
                            sequence = parts[0] if parts[0] else ""
                            transcript_id = parts[1]
                            sequences[transcript_id] = sequence
                
                # Be respectful to the server
                time.sleep(1)
                
            except Exception as e:
                print(f"Error querying batch {i//batch_size + 1}: {e}")
                # Add sequences with empty values for failed batch
                for transcript_id in batch:
                    if transcript_id not in sequences:
                        sequences[transcript_id] = ""
                continue
        
        return sequences
        
    except Exception as e:
        print(f"Error connecting to BioMart: {e}")
        return {}

def process_gnomad_data(input_file, output_file, fetch_sequences=True):
    """
    Process gnomAD constraint metrics data.
    
    Args:
        input_file: Path to input TSV file
        output_file: Path to output CSV file
        fetch_sequences: Whether to fetch protein sequences from Ensembl BioMart
    """
    print(f"Reading data from {input_file}...")
    
    # Read the TSV file
    df = pd.read_csv(input_file, sep='\t', low_memory=False)
    
    print(f"Initial data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Check if required columns exist
    required_cols = ['lof.oe_ci.upper', 'transcript_type', 'transcript']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Remove entries with missing LOEUF values (lof.oe_ci.upper)
    print("Removing entries with missing LOEUF values...")
    initial_count = len(df)
    df = df.dropna(subset=['lof.oe_ci.upper'])
    loeuf_filtered_count = len(df)
    print(f"Removed {initial_count - loeuf_filtered_count} entries with missing LOEUF values")
    
    # Keep only protein_coding transcripts
    print("Filtering for protein_coding transcripts...")
    df = df[df['transcript_type'] == 'protein_coding']
    protein_coding_count = len(df)
    print(f"Kept {protein_coding_count} protein_coding entries")
    
    # Remove duplicates based on transcript
    print("Removing duplicate transcripts...")
    df = df.drop_duplicates(subset=['transcript'])
    final_count = len(df)
    print(f"Removed {protein_coding_count - final_count} duplicate transcripts")
    
    # Keep only transcript and LOEUF columns
    print("Selecting only transcript and LOEUF columns...")
    df_final = df[['transcript', 'lof.oe_ci.upper']].copy()
    df_final.columns = ['transcript', 'LOEUF']
    
    # Fetch protein sequences if requested
    if fetch_sequences:
        print("\nFetching protein sequences from Ensembl BioMart...")
        transcript_list = df_final['transcript'].tolist()
        sequences = get_protein_sequences_biomart(transcript_list)
        
        print(f"Retrieved sequences for {len(sequences)} transcripts")
        
        # Add sequence column
        df_final['protein_sequence'] = df_final['transcript'].map(sequences)
        
        # Report statistics
        sequences_found = df_final['protein_sequence'].notna().sum()
        sequences_missing = len(df_final) - sequences_found
        print(f"Sequences found: {sequences_found}")
        print(f"Sequences missing: {sequences_missing}")
        
        if sequences_missing > 0:
            print(f"Warning: {sequences_missing} transcripts have no protein sequence")
    
    # Sort by transcript for consistent output
    df_final = df_final.sort_values('transcript')
    
    print(f"Final data shape: {df_final.shape}")
    
    # Save processed data
    print(f"Saving processed data to {output_file}...")
    df_final.to_csv(output_file, index=False)
    
    # Print summary statistics
    print("\nSummary:")
    print(f"Total protein_coding transcripts: {len(df_final)}")
    print(f"LOEUF range: {df_final['LOEUF'].min():.4f} - {df_final['LOEUF'].max():.4f}")
    
    return df_final

if __name__ == "__main__":
    import sys
    
    input_file = "data/train/gnomad.v4.1.constraint_metrics.tsv"
    output_file = "data/train/gnomad_processed.csv"
    
    # Check if user wants to fetch sequences (can be slow for large datasets)
    fetch_sequences = True
    if len(sys.argv) > 1 and sys.argv[1] == "--no-sequences":
        fetch_sequences = False
        print("Skipping sequence fetching (use without --no-sequences to fetch sequences)")
    
    try:
        processed_df = process_gnomad_data(input_file, output_file, fetch_sequences=fetch_sequences)
        print(f"\nProcessing completed successfully!")
        print(f"Output saved to: {output_file}")
    except Exception as e:
        print(f"Error processing data: {e}")