#!/usr/bin/env python3
"""
Script to extract missing and incomplete protein sequences from gnomad_processed.csv.
This script identifies transcripts with missing protein sequences (empty/NaN) or incomplete 
protein sequences (don't end with '*') and re-fetches them from Ensembl BioMart.
"""

import pandas as pd
import numpy as np
from biomart import BiomartServer
from typing import List, Dict
import time
import argparse
import sys

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
                            # BioMart returns: peptide \t transcript_id
                            peptide = parts[0].strip() if parts[0] else ""
                            transcript_id = parts[1].strip()
                            # Only store non-empty sequences
                            if peptide:
                                sequences[transcript_id] = peptide
                
                # Be respectful to the server
                time.sleep(1)
                
            except Exception as e:
                print(f"Error querying batch {i//batch_size + 1}: {e}")
                continue
        
        return sequences
        
    except Exception as e:
        print(f"Error connecting to BioMart: {e}")
        return {}

def extract_missing_proteins(input_file: str, output_file: str = None, batch_size: int = 100):
    """
    Extract missing and incomplete protein sequences from the processed gnomAD file.
    
    Args:
        input_file: Path to the gnomad_processed.csv file
        output_file: Path to save the updated file (if None, overwrites input)
        batch_size: Number of transcripts to query per batch
    """
    print(f"Reading data from {input_file}...")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    print(f"Total transcripts: {len(df)}")
    
    # Find transcripts with missing or incomplete protein sequences
    # Missing: empty, NaN, or 'nan'
    # Incomplete: non-empty sequences that don't end with '*'
    missing_mask = df['protein_sequence'].isna() | (df['protein_sequence'] == '') | (df['protein_sequence'] == 'nan')
    incomplete_mask = (~missing_mask) & (~df['protein_sequence'].str.endswith('*', na=False))
    
    # Combine both conditions
    needs_update_mask = missing_mask | incomplete_mask
    transcripts_to_update = df[needs_update_mask]['transcript'].tolist()
    
    missing_count = missing_mask.sum()
    incomplete_count = incomplete_mask.sum()
    
    print(f"Found {missing_count} transcripts with missing protein sequences")
    print(f"Found {incomplete_count} transcripts with incomplete protein sequences (don't end with '*')")
    print(f"Total transcripts needing update: {len(transcripts_to_update)}")
    
    if len(transcripts_to_update) == 0:
        print("No missing or incomplete protein sequences found. Nothing to do.")
        return
    
    # Show some examples of transcripts needing update
    print(f"Examples of transcripts needing update: {transcripts_to_update[:10]}")
    
    # Fetch protein sequences for transcripts needing update
    print("\nFetching protein sequences from Ensembl BioMart...")
    sequences = get_protein_sequences_biomart(transcripts_to_update, batch_size=batch_size)
    
    print(f"Successfully retrieved {len(sequences)} protein sequences")
    
    # Update the dataframe with new sequences
    updated_count = 0
    for transcript_id, sequence in sequences.items():
        if sequence:  # Only update if we got a non-empty sequence
            mask = df['transcript'] == transcript_id
            if mask.any():
                df.loc[mask, 'protein_sequence'] = sequence
                updated_count += 1
    
    print(f"Updated {updated_count} protein sequences in the dataframe")
    
    # Calculate final statistics
    still_missing_mask = df['protein_sequence'].isna() | (df['protein_sequence'] == '') | (df['protein_sequence'] == 'nan')
    still_incomplete_mask = (~still_missing_mask) & (~df['protein_sequence'].str.endswith('*', na=False))
    still_needs_update_mask = still_missing_mask | still_incomplete_mask
    
    still_missing_count = still_missing_mask.sum()
    still_incomplete_count = still_incomplete_mask.sum()
    total_still_problematic = still_needs_update_mask.sum()
    
    print(f"Remaining missing sequences: {still_missing_count}")
    print(f"Remaining incomplete sequences: {still_incomplete_count}")
    print(f"Successfully filled sequences: {len(transcripts_to_update) - total_still_problematic}")
    
    # Save the updated file
    if output_file is None:
        output_file = input_file
    
    print(f"Saving updated data to {output_file}...")
    df.to_csv(output_file, index=False)
    
    print("\nSummary:")
    print(f"Total transcripts: {len(df)}")
    print(f"Originally missing sequences: {missing_count}")
    print(f"Originally incomplete sequences: {incomplete_count}")
    print(f"Total transcripts needing update: {len(transcripts_to_update)}")
    print(f"Successfully retrieved sequences: {len(sequences)}")
    print(f"Updated sequences in file: {updated_count}")
    print(f"Still missing sequences: {still_missing_count}")
    print(f"Still incomplete sequences: {still_incomplete_count}")
    
    if total_still_problematic > 0:
        print("\nTranscripts still with problematic sequences:")
        if still_missing_count > 0:
            print("  Missing sequences:")
            still_missing_transcripts = df[still_missing_mask]['transcript'].tolist()
            for transcript in still_missing_transcripts[:10]:
                print(f"    {transcript}")
            if len(still_missing_transcripts) > 10:
                print(f"    ... and {len(still_missing_transcripts) - 10} more missing")
        
        if still_incomplete_count > 0:
            print("  Incomplete sequences (don't end with '*'):")
            still_incomplete_transcripts = df[still_incomplete_mask]['transcript'].tolist()
            for transcript in still_incomplete_transcripts[:10]:
                print(f"    {transcript}")
            if len(still_incomplete_transcripts) > 10:
                print(f"    ... and {len(still_incomplete_transcripts) - 10} more incomplete")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Extract missing and incomplete protein sequences from gnomad_processed.csv')
    parser.add_argument('input_file', help='Path to the gnomad_processed.csv file')
    parser.add_argument('-o', '--output', help='Output file path (default: overwrite input file)')
    parser.add_argument('-b', '--batch-size', type=int, default=100, help='Batch size for BioMart queries (default: 100)')
    parser.add_argument('--dry-run', action='store_true', help='Only analyze missing and incomplete sequences without fetching')
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("DRY RUN MODE: Analyzing missing and incomplete sequences only")
        df = pd.read_csv(args.input_file)
        missing_mask = df['protein_sequence'].isna() | (df['protein_sequence'] == '') | (df['protein_sequence'] == 'nan')
        incomplete_mask = (~missing_mask) & (~df['protein_sequence'].str.endswith('*', na=False))
        needs_update_mask = missing_mask | incomplete_mask
        
        missing_count = missing_mask.sum()
        incomplete_count = incomplete_mask.sum()
        total_needing_update = needs_update_mask.sum()
        
        print(f"Total transcripts: {len(df)}")
        print(f"Missing protein sequences: {missing_count}")
        print(f"Incomplete protein sequences (don't end with '*'): {incomplete_count}")
        print(f"Total needing update: {total_needing_update}")
        print(f"Percentage missing: {missing_count/len(df)*100:.1f}%")
        print(f"Percentage incomplete: {incomplete_count/len(df)*100:.1f}%")
        print(f"Percentage needing update: {total_needing_update/len(df)*100:.1f}%")
        return
    
    try:
        extract_missing_proteins(args.input_file, args.output, args.batch_size)
        print(f"\nExtraction completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()