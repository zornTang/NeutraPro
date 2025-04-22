"""
Script for:
1) extracting neutrophil Ensembl gene IDs from a TSV,
2) parsing MANE_Select / Ensembl_canonical transcript IDs from GTF,
3) extracting corresponding protein sequences from a FASTA,
4) filtering those proteins by a gene‑ID list.
"""

import argparse
import csv
import gzip
import logging
import re
import sys
from pathlib import Path
from typing import Set, Tuple

from Bio import SeqIO

# Constants for regex patterns and FASTA header parsing
REGEX_GENE_ID       = re.compile(r'gene_id "([^"]+)"')
REGEX_TRANSCRIPT_ID = re.compile(r'transcript_id "([^"]+)"')
HEADER_SEP          = "|"


def setup_logging(level: str = "INFO") -> None:
    """
    Configure root logger.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def open_maybe_gz(path: Path, mode: str = "rt"):
    """
    Open a file, using gzip.open if it ends with .gz/.gzip.
    """
    if path.suffix in (".gz", ".gzip"):
        return gzip.open(path, mode)
    return path.open(mode)


def parse_neutrophil_ids(tsv_path: Path, out_path: Path) -> Set[str]:
    """
    Read TSV, extract the 'Ensembl' column into a set of gene IDs,
    write them (one per line) to out_path, and return the set.
    """
    gene_ids: Set[str] = set()
    try:
        with tsv_path.open(newline="") as fin, out_path.open("w") as fout:
            reader = csv.DictReader(fin, delimiter="\t")
            for row in reader:
                gid = row.get("Ensembl")
                if gid:
                    fout.write(gid + "\n")
                    gene_ids.add(gid)
        logging.info("Extracted %d neutrophil gene IDs to %s", len(gene_ids), out_path)
    except Exception as e:
        logging.error("Failed to parse neutrophil IDs: %s", e)
        sys.exit(1)
    return gene_ids


def parse_mane_transcripts(gtf_path: Path) -> Tuple[Set[str], Set[str]]:
    """
    Scan a GTF (possibly gzipped) for MANE_Select and Ensembl_canonical tags.
    Return two sets:
      - full_ids: transcript IDs with version
      - base_ids: transcript IDs without version
    """
    gene_to_mane = {}
    gene_to_canon = {}

    try:
        with open_maybe_gz(gtf_path, "rt") as fin:
            for line in fin:
                if line.startswith("#") or "\ttranscript\t" not in line:
                    continue
                if 'tag "MANE_Select"' in line:
                    gid = REGEX_GENE_ID.search(line).group(1)
                    tid = REGEX_TRANSCRIPT_ID.search(line).group(1)
                    gene_to_mane[gid] = tid
                elif 'tag "Ensembl_canonical"' in line:
                    gid = REGEX_GENE_ID.search(line).group(1)
                    tid = REGEX_TRANSCRIPT_ID.search(line).group(1)
                    gene_to_canon.setdefault(gid, tid)
    except Exception as e:
        logging.error("Failed to parse GTF: %s", e)
        sys.exit(1)

    # Merge with MANE_Select priority, fallback to canonical
    full_ids = set(gene_to_mane.values()) | {
        tid for gid, tid in gene_to_canon.items() if gid not in gene_to_mane
    }
    base_ids = {tid.split(".", 1)[0] for tid in full_ids}
    logging.info("Parsed %d transcript IDs (with version), %d base IDs",
                 len(full_ids), len(base_ids))
    return full_ids, base_ids


def extract_mane_proteins(
    fasta_path: Path,
    out_fasta: Path,
    tids_full: Set[str],
    tids_base: Set[str]
) -> int:
    """
    Extract protein sequences whose transcript IDs (field #2 in header)
    match either full or base IDs. Returns the count written.
    """
    count = 0
    try:
        with open_maybe_gz(fasta_path, "rt") as fin, out_fasta.open("w") as fout:
            for rec in SeqIO.parse(fin, "fasta"):
                parts = rec.id.split(HEADER_SEP)
                if len(parts) < 2:
                    continue
                tid = parts[1]
                base = tid.split(".", 1)[0]
                if tid in tids_full or base in tids_base:
                    SeqIO.write(rec, fout, "fasta")
                    count += 1
        logging.info("Extracted %d MANE protein sequences to %s", count, out_fasta)
    except Exception as e:
        logging.error("Failed to extract MANE proteins: %s", e)
        sys.exit(1)
    return count


def filter_proteins_by_genes(
    input_fasta: Path,
    gene_ids: Set[str],
    out_fasta: Path
) -> int:
    """
    From input_fasta, write only records whose gene ID (field #3 in header)
    matches one of gene_ids. Returns the count written.
    """
    count = 0
    try:
        with input_fasta.open() as fin, out_fasta.open("w") as fout:
            for rec in SeqIO.parse(fin, "fasta"):
                parts = rec.id.split(HEADER_SEP)
                if len(parts) < 3:
                    continue
                gid = parts[2].split(".", 1)[0]
                if gid in gene_ids:
                    SeqIO.write(rec, fout, "fasta")
                    count += 1
        logging.info("Filtered %d proteins for %d genes to %s",
                     count, len(gene_ids), out_fasta)
    except Exception as e:
        logging.error("Failed to filter proteins by genes: %s", e)
        sys.exit(1)
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Extract and filter MANE protein sequences"
    )
    parser.add_argument(
        "--neut-tsv",
        type=Path,
        default=Path("../immune_cell_category_rna_neutrophil_Immune.tsv"),
        help="TSV file with neutrophil Ensembl gene IDs (default: %(default)s)"
    )
    parser.add_argument(
        "--gtf",
        type=Path,
        default=Path("../gencode.v47.annotation.gtf.gz"),
        help="GTF file (gzipped or plain) with MANE tags (default: %(default)s)"
    )
    parser.add_argument(
        "--pc-fasta",
        type=Path,
        default=Path("../gencode.v47.pc_translations.fa.gz"),
        help="Protein‑coding FASTA (gzipped or plain) (default: %(default)s)"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output"),
        help="Output directory for intermediate and final files (default: %(default)s)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging level (default: %(default)s)"
    )
    args = parser.parse_args()

    setup_logging(args.log_level)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: extract neutrophil gene IDs
    neut_out = args.out_dir / "neutrophil_gene_ids.txt"
    neut_ids = parse_neutrophil_ids(args.neut_tsv, neut_out)

    # Step 2: parse MANE transcript IDs
    mane_full, mane_base = parse_mane_transcripts(args.gtf)

    # Step 3: extract MANE protein sequences
    mane_fasta = args.out_dir / "mane_proteins.fa"
    extract_mane_proteins(args.pc_fasta, mane_fasta, mane_full, mane_base)

    # Step 4: filter by neutrophil gene IDs
    filtered = args.out_dir / "neutrophil_mane_proteins.fa"
    filter_proteins_by_genes(mane_fasta, neut_ids, filtered)


if __name__ == "__main__":
    main()