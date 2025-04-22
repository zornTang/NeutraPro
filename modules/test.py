#!/usr/bin/env python3
import gzip
import re
from Bio import SeqIO
from itertools import islice

# —— 1. 输入/输出文件路径 —— #
gtf_path     = "../gencode.v47.annotation.gtf.gz"
fasta_path   = "../gencode.v47.pc_translations.fa.gz"
output_fasta = "../mane_proteins.fa"

# —— 2. 从 GTF 中解析 MANE_Select / Ensembl_canonical 转录本 —— #
pattern_gene = re.compile(r'gene_id "([^"]+)"')
pattern_tid  = re.compile(r'transcript_id "([^"]+)"')

gene_mane      = {}  # gene_id -> transcript_id（MANE_Select）
gene_canonical = {}  # gene_id -> transcript_id（Ensembl_canonical）

with gzip.open(gtf_path, 'rt') as gtf:
    for line in gtf:
        if line.startswith("#") or "\ttranscript\t" not in line:
            continue

        if 'tag "MANE_Select"' in line:
            g = pattern_gene.search(line).group(1)
            t = pattern_tid.search(line).group(1)
            gene_mane[g] = t

        elif 'tag "Ensembl_canonical"' in line:
            g = pattern_gene.search(line).group(1)
            t = pattern_tid.search(line).group(1)
            # 仅当该基因尚无 MANE_Select 时使用 canonical 作为后备
            if g not in gene_mane:
                gene_canonical[g] = t

# 合并优先级：MANE_Select > Ensembl_canonical
selected_tids = set(gene_mane.values()) | set(gene_canonical.values())
selected_base = {tid.split('.')[0] for tid in selected_tids}
print(f"Selected {len(selected_tids)} transcripts (with versions), "
      f"{len(selected_base)} base IDs (no versions).")

# —— 3. Inspect and extract —— #
with gzip.open(fasta_path, 'rt') as fasta_in, open(output_fasta, 'w') as fasta_out:
    # 先看前 5 条，确认 rec.id 和 rec.description
    for rec in islice(SeqIO.parse(fasta_in, "fasta"), 5):
        print("HEADER:", rec.id, rec.description)
    fasta_in.seek(0)

    cnt = 0
    for rec in SeqIO.parse(fasta_in, "fasta"):
        # header 示例：ENSP...|ENST00000618323.5|ENSG...|...
        parts = rec.id.split('|')
        if len(parts) < 2:
            continue
        tid_full = parts[1]              # 带版本号的 transcript_id
        tid_base = tid_full.split('.')[0]

        if tid_full in selected_tids or tid_base in selected_base:
            # print(f"MATCHED: {tid_full}")      # 验证匹配
            SeqIO.write(rec, fasta_out, "fasta")
            cnt += 1

    print(f"Written {cnt} representative proteins to {output_fasta}.")
