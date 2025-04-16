import argparse
import time
from pathlib import Path
import logging
import datetime

import torch
import h5py
from transformers import T5EncoderModel, T5Tokenizer

# Configure logging
log_filename = f"embedding_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=log_filename,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Starting the embedding process...")

# Set device for computation
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")


def load_t5_model(model_dir=None, transformer_link="Rostlab/prot_t5_xl_half_uniref50-enc"):
    """
    Load the T5 model and tokenizer.

    Args:
        model_dir (str): Path to the directory containing the cached model.
        transformer_link (str): Hugging Face model identifier.

    Returns:
        model (T5EncoderModel): Pre-trained T5 model.
        tokenizer (T5Tokenizer): Tokenizer for the T5 model.
    """
    logging.info(f"Loading model: {transformer_link}")
    if model_dir:
        logging.info(f"Loading cached model from: {model_dir}")

    model = T5EncoderModel.from_pretrained(transformer_link, cache_dir=model_dir)
    if device == torch.device("cpu"):
        logging.info("Casting model to full precision for CPU...")
        model.to(torch.float32)

    model = model.to(device).eval()
    tokenizer = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False)
    return model, tokenizer


def read_fasta_file(fasta_path):
    """
    Read a FASTA file and return a dictionary of sequences.

    Args:
        fasta_path (str): Path to the FASTA file.

    Returns:
        dict: Dictionary with sequence IDs as keys and sequences as values.
    """
    sequences = {}
    try:
        with open(fasta_path, 'r') as fasta_file:
            for line in fasta_file:
                if line.startswith('>'):
                    seq_id = line[1:].strip()  # Header is assumed to be a numeric ID
                    sequences[seq_id] = ""
                else:
                    sequences[seq_id] += line.strip()  # Directly append the sequence
        logging.info(f"Loaded {len(sequences)} sequences from {fasta_path}")
    except Exception as e:
        logging.error(f"Error reading FASTA file: {e}")
        raise
    return sequences


def process_sequences(seq_dict, model, tokenizer, emb_path, per_protein, max_residues, max_seq_len, max_batch):
    """
    Generate embeddings for sequences and save them to an HDF5 file.

    Args:
        seq_dict (dict): Dictionary of sequences.
        model (T5EncoderModel): Pre-trained T5 model.
        tokenizer (T5Tokenizer): Tokenizer for the T5 model.
        emb_path (str): Path to save the embeddings.
        per_protein (bool): Whether to return mean-pooled embeddings.
        max_residues (int): Maximum number of residues per batch.
        max_seq_len (int): Maximum sequence length for batch processing.
        max_batch (int): Maximum number of sequences per batch.

    Returns:
        None
    """
    logging.info(f"Total sequences: {len(seq_dict)}")
    avg_length = sum(len(seq) for seq in seq_dict.values()) / len(seq_dict)
    n_long = sum(1 for seq in seq_dict.values() if len(seq) > max_seq_len)
    logging.info(f"Average sequence length: {avg_length}")
    logging.info(f"Number of sequences > {max_seq_len}: {n_long}")

    # Sort sequences by length (descending)
    seq_dict = sorted(seq_dict.items(), key=lambda x: len(x[1]), reverse=True)

    start_time = time.time()
    embeddings = {}

    batch = []
    for seq_idx, (seq_id, seq) in enumerate(seq_dict, 1):
        seq_len = len(seq)
        seq = ' '.join(list(seq))  # Add spaces between amino acids
        batch.append((seq_id, seq, seq_len))

        # Check if batch exceeds limits
        n_res_batch = sum(s_len for _, _, s_len in batch) + seq_len
        if len(batch) >= max_batch or n_res_batch >= max_residues or seq_idx == len(seq_dict) or seq_len > max_seq_len:
            process_batch(batch, model, tokenizer, embeddings, per_protein)
            batch = []

    # Save embeddings to HDF5
    save_embeddings_to_hdf5(embeddings, emb_path)

    end_time = time.time()
    logging.info(f"Total embeddings: {len(embeddings)}")
    logging.info(f"Total time: {end_time - start_time:.2f}s; Time per protein: {(end_time - start_time) / len(embeddings):.4f}s")


def process_batch(batch, model, tokenizer, embeddings, per_protein):
    """
    Process a batch of sequences and generate embeddings.

    Args:
        batch (list): List of tuples (seq_id, seq, seq_len).
        model (T5EncoderModel): Pre-trained T5 model.
        tokenizer (T5Tokenizer): Tokenizer for the T5 model.
        embeddings (dict): Dictionary to store embeddings.
        per_protein (bool): Whether to return mean-pooled embeddings.

    Returns:
        None
    """
    seq_ids, seqs, seq_lens = zip(*batch)
    token_encoding = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(token_encoding['input_ids']).to(device)
    attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

    try:
        with torch.no_grad():
            embedding_repr = model(input_ids, attention_mask=attention_mask)
    except RuntimeError as e:
        logging.error(f"RuntimeError during embedding: {e}")
        return

    for idx, seq_id in enumerate(seq_ids):
        seq_len = seq_lens[idx]
        emb = embedding_repr.last_hidden_state[idx, :seq_len]
        if per_protein:
            emb = emb.mean(dim=0)
        embeddings[seq_id] = emb.cpu().numpy()

        # Log the progress of each sequence
        logging.info(f"Processed sequence ID: {seq_id}, Length: {seq_len}")


def save_embeddings_to_hdf5(embeddings, emb_path):
    """
    Save embeddings to an HDF5 file.

    Args:
        embeddings (dict): Dictionary of embeddings.
        emb_path (str): Path to save the HDF5 file.

    Returns:
        None
    """
    try:
        with h5py.File(emb_path, "w") as hdf5_file:
            for seq_id, embedding in embeddings.items():
                hdf5_file.create_dataset(seq_id, data=embedding)
        logging.info(f"Embeddings saved to: {emb_path}")
    except Exception as e:
        logging.error(f"Error saving embeddings to HDF5: {e}")
        raise


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Generate T5 embeddings for protein sequences in FASTA format.")
    parser.add_argument('--fasta_path', required=False, type=str, default="/data/qin2/chein/NeutraPro/data/human_data_balanced.fasta",
                        help="Path to the input FASTA file. Default: '/data/qin2/chein/NeutraPro/data/human_data_balanced.fasta'")
    parser.add_argument('--save_path', required=False, type=str, default="/data/qin2/chein/NeutraPro/data/embeddings.h5",
                        help="Path to save the embeddings. Default: '/data/qin2/chein/NeutraPro/data/embeddings.h5'")
    parser.add_argument('--model', type=str, default="/data/qin2/chein/NeutraPro/pretrain_model",
                        help="Path to the cached model directory. Default: '/data/qin2/chein/NeutraPro/pretrain_model'")
    parser.add_argument('--per_protein', type=int, default=1,
                        help="Return mean-pooled embeddings (1) or per-residue embeddings (0). Default: 1")
    return parser.parse_args()


def main():
    """
    Main function to execute the embedding generation process.
    """
    args = parse_arguments()
    seq_path = Path(args.fasta_path)
    emb_path = Path(args.save_path)
    model_dir = Path(args.model) if args.model else None
    per_protein = bool(args.per_protein)

    seq_dict = read_fasta_file(seq_path)
    model, tokenizer = load_t5_model(model_dir)
    process_sequences(seq_dict, model, tokenizer, emb_path, per_protein, max_residues=4000, max_seq_len=1000, max_batch=100)


if __name__ == '__main__':
    main()