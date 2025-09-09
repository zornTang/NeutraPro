#!/usr/bin/env python3
"""
Script to clean protein sequence data by removing blank sequences, duplicates, and incomplete sequences.
This script removes rows with missing/empty protein sequences (including 'Sequence unavailable'), 
duplicate protein sequences, and optionally incomplete sequences (those that don't end with '*').
"""

import pandas as pd
import numpy as np
import argparse
import sys

def clean_protein_data(input_file: str, output_file: str = None, remove_blanks: bool = True, remove_duplicates: bool = True, remove_incomplete: bool = False):
    """
    Clean protein sequence data by removing blank sequences, duplicates, and/or incomplete sequences.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to save the cleaned file (if None, overwrites input)
        remove_blanks: Whether to remove rows with blank protein sequences
        remove_duplicates: Whether to remove duplicate protein sequences
        remove_incomplete: Whether to remove incomplete sequences (don't end with '*')
    """
    print(f"Reading data from {input_file}...")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    print(f"Original dataset: {len(df)} rows")
    
    initial_count = len(df)
    
    # Remove blank protein sequences
    if remove_blanks:
        print("\nRemoving blank protein sequences...")
        # Find rows with blank protein sequences (empty, NaN, 'nan', or 'Sequence unavailable')
        blank_mask = (df['protein_sequence'].isna() | 
                     (df['protein_sequence'] == '') | 
                     (df['protein_sequence'] == 'nan') |
                     (df['protein_sequence'] == 'Sequence unavailable'))
        blank_count = blank_mask.sum()
        
        print(f"Found {blank_count} rows with blank protein sequences")
        
        if blank_count > 0:
            # Show some examples of blank sequences
            blank_examples = df[blank_mask][['transcript', 'protein_sequence']].head(5)
            print("Examples of rows with blank sequences:")
            for _, row in blank_examples.iterrows():
                print(f"  Transcript: {row['transcript']}, Sequence: '{row['protein_sequence']}'")
            
            # Remove blank sequences
            df = df[~blank_mask].copy()
            print(f"Removed {blank_count} rows with blank sequences")
            print(f"Remaining rows: {len(df)}")
    
    # Remove duplicate protein sequences
    if remove_duplicates:
        print("\nRemoving duplicate protein sequences...")
        
        # Count duplicates before removal
        duplicate_mask = df.duplicated(subset=['protein_sequence'], keep=False)
        duplicate_count = duplicate_mask.sum()
        unique_sequences_before = df['protein_sequence'].nunique()
        
        print(f"Found {duplicate_count} rows with duplicate protein sequences")
        print(f"Unique protein sequences before deduplication: {unique_sequences_before}")
        
        if duplicate_count > 0:
            # Show some examples of duplicated sequences
            duplicated_sequences = df[duplicate_mask]['protein_sequence'].value_counts().head(3)
            print("Top 3 most frequent duplicate sequences:")
            for seq, count in duplicated_sequences.items():
                seq_preview = seq[:50] + "..." if len(seq) > 50 else seq
                print(f"  Sequence: {seq_preview} (appears {count} times)")
            
            # Remove duplicates, keeping the first occurrence
            df_before_dedup = df.copy()
            df = df.drop_duplicates(subset=['protein_sequence'], keep='first').copy()
            removed_duplicates = len(df_before_dedup) - len(df)
            
            print(f"Removed {removed_duplicates} duplicate rows")
            print(f"Remaining rows: {len(df)}")
            
            unique_sequences_after = df['protein_sequence'].nunique()
            print(f"Unique protein sequences after deduplication: {unique_sequences_after}")
    
    # Count and optionally remove incomplete sequences (don't end with '*')
    print("\nAnalyzing incomplete protein sequences...")
    # Only analyze non-blank sequences for completeness
    non_blank_mask = ~(df['protein_sequence'].isna() | 
                      (df['protein_sequence'] == '') | 
                      (df['protein_sequence'] == 'nan') |
                      (df['protein_sequence'] == 'Sequence unavailable'))
    incomplete_mask = non_blank_mask & (~df['protein_sequence'].str.endswith('*', na=False))
    incomplete_count = incomplete_mask.sum()
    
    print(f"Found {incomplete_count} sequences that don't end with '*' ({incomplete_count/len(df)*100:.1f}% of total)")
    
    if incomplete_count > 0:
        # Show some examples of incomplete sequences
        incomplete_examples = df[incomplete_mask][['transcript', 'protein_sequence']].head(5)
        print("Examples of incomplete sequences:")
        for _, row in incomplete_examples.iterrows():
            seq = str(row['protein_sequence'])
            seq_preview = seq[:50] + "..." if len(seq) > 50 else seq
            print(f"  Transcript: {row['transcript']}, Sequence: {seq_preview}")
        
        if remove_incomplete:
            print(f"Removing {incomplete_count} incomplete sequences...")
            df = df[~incomplete_mask].copy()
            print(f"Removed {incomplete_count} incomplete sequences")
            print(f"Remaining rows: {len(df)}")
        else:
            print("Note: Use --remove-incomplete flag to remove these sequences")
    
    # Final statistics
    final_count = len(df)
    total_removed = initial_count - final_count
    
    print(f"\nFinal statistics:")
    print(f"Original rows: {initial_count}")
    print(f"Final rows: {final_count}")
    print(f"Total removed: {total_removed}")
    print(f"Retention rate: {final_count/initial_count*100:.1f}%")
    
    # Save the cleaned file
    if output_file is None:
        output_file = input_file
    
    print(f"\nSaving cleaned data to {output_file}...")
    df.to_csv(output_file, index=False)
    
    print("Data cleaning completed successfully!")
    
    return df

def classify_terminator_codons(df: pd.DataFrame):
    """
    Classify sequences based on presence of terminator codons (*) and remove all stop codons.
    
    Args:
        df: DataFrame containing protein sequences
        
    Returns:
        df: DataFrame with terminator classification and cleaned sequences
    """
    print("\nClassifying sequences by terminator codon presence...")
    
    # Create classification column
    # 1 = has terminator codon (*), 0 = no terminator codon
    df['has_terminator'] = df['protein_sequence'].str.contains('*', na=False, regex=False).astype(int)
    
    # Add descriptive label
    df['terminator_label'] = df['has_terminator'].map({1: 'with_terminator', 0: 'without_terminator'})
    
    # Count statistics before cleaning
    with_terminator = (df['has_terminator'] == 1).sum()
    without_terminator = (df['has_terminator'] == 0).sum()
    
    print(f"Sequences with terminator codon (*): {with_terminator} ({with_terminator/len(df)*100:.1f}%)")
    print(f"Sequences without terminator codon: {without_terminator} ({without_terminator/len(df)*100:.1f}%)")
    
    # Remove all stop codons from sequences
    print("Removing all stop codons (*) from protein sequences...")
    df['protein_sequence_cleaned'] = df['protein_sequence'].str.replace('*', '', regex=False)
    
    # Show sequence length changes
    original_lengths = df['protein_sequence'].str.len()
    cleaned_lengths = df['protein_sequence_cleaned'].str.len()
    removed_codons = original_lengths - cleaned_lengths
    
    print(f"Stop codons removed per sequence - Mean: {removed_codons.mean():.2f}, Max: {removed_codons.max()}")
    
    return df

def classify_proteins_by_loeuf(input_file: str, output_file: str = None, loeuf_threshold: float = 0.6):
    """
    Classify proteins as essential or non-essential based on LOEUF values.
    
    Args:
        input_file: Path to the input CSV file containing LOEUF data
        output_file: Path to save the classified file (if None, overwrites input)
        loeuf_threshold: Threshold for classification (default: 0.6)
                        Values < threshold = essential (1)
                        Values >= threshold = non-essential (0)
    """
    print(f"Reading data from {input_file}...")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    print(f"Dataset contains {len(df)} rows")
    
    # Check if LOEUF column exists
    if 'LOEUF' not in df.columns:
        raise ValueError("The CSV file does not contain a 'LOEUF' column.")
    
    print(f"Using LOEUF threshold: {loeuf_threshold}")
    print(f"Classification rule: LOEUF < {loeuf_threshold} = essential, LOEUF >= {loeuf_threshold} = non-essential")
    
    # Add protein classification column
    # LOEUF < threshold = essential (1), LOEUF >= threshold = non-essential (0)
    df['protein_essentiality'] = (df['LOEUF'] < loeuf_threshold).astype(int)
    
    # Add descriptive label
    df['essentiality_label'] = df['protein_essentiality'].map({1: 'essential', 0: 'non-essential'})
    
    # Print classification statistics
    essential_count = (df['protein_essentiality'] == 1).sum()
    non_essential_count = (df['protein_essentiality'] == 0).sum()
    
    print(f"\nClassification results:")
    print(f"Essential proteins (LOEUF < {loeuf_threshold}): {essential_count} ({essential_count/len(df)*100:.1f}%)")
    print(f"Non-essential proteins (LOEUF >= {loeuf_threshold}): {non_essential_count} ({non_essential_count/len(df)*100:.1f}%)")
    print(f"Total proteins: {len(df)}")
    
    # Display LOEUF value distribution
    print(f"\nLOEUF value statistics:")
    print(f"Min LOEUF: {df['LOEUF'].min():.3f}")
    print(f"Max LOEUF: {df['LOEUF'].max():.3f}")
    print(f"Mean LOEUF: {df['LOEUF'].mean():.3f}")
    print(f"Median LOEUF: {df['LOEUF'].median():.3f}")
    
    # Save the classified data
    if output_file is None:
        output_file = input_file.replace('.csv', '_classified.csv')
    
    print(f"\nSaving classified data to {output_file}...")
    df.to_csv(output_file, index=False)
    
    # Display sample of classified data
    print(f"\nSample of classified data:")
    columns_to_show = ['transcript', 'LOEUF', 'protein_essentiality', 'essentiality_label']
    available_columns = [col for col in columns_to_show if col in df.columns]
    print(df[available_columns].head(10).to_string(index=False))
    
    print(f"\nProtein classification completed successfully!")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Clean protein sequence data and classify proteins by LOEUF values')
    parser.add_argument('input_file', help='Path to the input CSV file')
    parser.add_argument('-o', '--output', help='Output file path (default: overwrite input file)')
    parser.add_argument('--no-remove-blanks', action='store_true', help='Skip removing blank protein sequences')
    parser.add_argument('--no-remove-duplicates', action='store_true', help='Skip removing duplicate protein sequences')
    parser.add_argument('--remove-incomplete', action='store_true', help='Remove incomplete sequences (don\'t end with \'*\')')
    parser.add_argument('--dry-run', action='store_true', help='Only analyze data without making changes')
    parser.add_argument('--classify-loeuf', action='store_true', help='Classify proteins by LOEUF values (essential vs non-essential)')
    parser.add_argument('--loeuf-threshold', type=float, default=0.6, help='LOEUF threshold for classification (default: 0.6)')
    parser.add_argument('--loeuf-only', action='store_true', help='Only perform LOEUF classification without data cleaning')
    parser.add_argument('--classify-terminator', action='store_true', help='Classify sequences by terminator codon presence and remove stop codons')
    parser.add_argument('--terminator-only', action='store_true', help='Only perform terminator classification without data cleaning')
    
    args = parser.parse_args()
    
    remove_blanks = not args.no_remove_blanks
    remove_duplicates = not args.no_remove_duplicates
    remove_incomplete = args.remove_incomplete
    
    # If only LOEUF classification is requested
    if args.loeuf_only:
        try:
            classify_proteins_by_loeuf(args.input_file, args.output, args.loeuf_threshold)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
        return
    
    # If only terminator classification is requested
    if args.terminator_only:
        try:
            df = pd.read_csv(args.input_file)
            df = classify_terminator_codons(df)
            
            output_file = args.output if args.output else args.input_file.replace('.csv', '_terminator_classified.csv')
            print(f"Saving terminator-classified data to {output_file}...")
            df.to_csv(output_file, index=False)
            print("Terminator classification completed successfully!")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
        return
    
    if args.dry_run:
        print("DRY RUN MODE: Analyzing data without making changes")
        df = pd.read_csv(args.input_file)
        
        print(f"Total rows: {len(df)}")
        
        # Analyze blank sequences
        blank_mask = (df['protein_sequence'].isna() | 
                     (df['protein_sequence'] == '') | 
                     (df['protein_sequence'] == 'nan') |
                     (df['protein_sequence'] == 'Sequence unavailable'))
        blank_count = blank_mask.sum()
        print(f"Blank protein sequences: {blank_count} ({blank_count/len(df)*100:.1f}%)")
        
        # Analyze duplicates
        duplicate_mask = df.duplicated(subset=['protein_sequence'], keep=False)
        duplicate_count = duplicate_mask.sum()
        unique_sequences = df['protein_sequence'].nunique()
        print(f"Duplicate protein sequences: {duplicate_count} rows ({duplicate_count/len(df)*100:.1f}%)")
        print(f"Unique protein sequences: {unique_sequences}")
        
        # Analyze incomplete sequences
        non_blank_mask = ~blank_mask
        incomplete_mask = non_blank_mask & (~df['protein_sequence'].str.endswith('*', na=False))
        incomplete_count = incomplete_mask.sum()
        print(f"Incomplete sequences (don't end with '*'): {incomplete_count} ({incomplete_count/len(df)*100:.1f}%)")
        
        # Estimate final count
        df_temp = df.copy()
        if remove_blanks:
            df_temp = df_temp[~blank_mask]
        if remove_duplicates:
            df_temp = df_temp.drop_duplicates(subset=['protein_sequence'], keep='first')
        if remove_incomplete:
            # Recalculate incomplete mask for the potentially reduced dataset
            non_blank_temp = ~(df_temp['protein_sequence'].isna() | 
                              (df_temp['protein_sequence'] == '') | 
                              (df_temp['protein_sequence'] == 'nan') |
                              (df_temp['protein_sequence'] == 'Sequence unavailable'))
            incomplete_temp = non_blank_temp & (~df_temp['protein_sequence'].str.endswith('*', na=False))
            df_temp = df_temp[~incomplete_temp]
        
        estimated_final = len(df_temp)
        estimated_removed = len(df) - estimated_final
        
        print(f"\nEstimated results:")
        print(f"Rows that would be removed: {estimated_removed}")
        print(f"Rows that would remain: {estimated_final}")
        print(f"Estimated retention rate: {estimated_final/len(df)*100:.1f}%")
        return
    
    try:
        # Perform data cleaning first
        df_cleaned = clean_protein_data(args.input_file, args.output, remove_blanks, remove_duplicates, remove_incomplete)
        
        # If terminator classification is requested
        if args.classify_terminator:
            print("\n" + "="*50)
            print("Starting terminator codon classification...")
            df_cleaned = classify_terminator_codons(df_cleaned)
        
        # If LOEUF classification is also requested
        if args.classify_loeuf:
            print("\n" + "="*50)
            print("Starting LOEUF classification...")
            
            # Use the cleaned data for classification
            output_file = args.output if args.output else args.input_file
            final_output = output_file.replace('.csv', '_classified.csv')
            
            classify_proteins_by_loeuf(output_file, final_output, args.loeuf_threshold)
        elif args.classify_terminator:
            # Save the terminator-classified data
            output_file = args.output if args.output else args.input_file
            final_output = output_file.replace('.csv', '_terminator_classified.csv')
            print(f"\nSaving terminator-classified data to {final_output}...")
            df_cleaned.to_csv(final_output, index=False)
            print("Terminator classification completed successfully!")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()