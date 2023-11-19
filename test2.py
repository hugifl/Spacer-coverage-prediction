import argparse
import numpy 
import pandas as pd
from utils_sequence import parse_fasta, one_hot_encode, extract_sequences_and_sequence_info

fasta_file = '../exon_coverage_input_output/U00096.3.fasta'
coverage_df = pd.read_csv('../exon_coverage_input_output/output/test.csv')
operon_df = pd.read_csv('../exon_coverage_input_output/operon_test.tsv', sep='\t')
gene_df = pd.read_csv('../exon_coverage_input_output/gene_test.tsv', sep='\t')
genome = parse_fasta(fasta_file)
window_size = 100  

encoded_sequences = extract_sequences_and_sequence_info(coverage_df, genome, window_size, operon_df, gene_df)

print(encoded_sequences[0])
print("twowowoowwowo")
print(encoded_sequences[1])