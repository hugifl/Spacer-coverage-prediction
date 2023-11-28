import argparse
import numpy as np
import pandas as pd
from utils_sequence import parse_fasta, one_hot_encode, extract_sequences_and_sequence_info
from sklearn.model_selection import train_test_split
from utils_coverage import get_gene_spacer_counts, normalize_coverage_per_operon, normalize_coverage_per_gene, replace_ones, dataframe_to_2darray, get_operon_spacer_counts
from utils_coverage import lowest_expressed_genes, filter_windows, get_windows, define_peaks, custom_train_test_split
from data_loading import data_loading
## Load your data
#data = np.load('../exon_coverage_input_output/output/XY_data_2000_1000.npz')
#outdir = '../exon_coverage_input_output/output/'
#window_size = 2000
#overlap_size = 0
#
#a = np.arange(20).reshape(-1, 1)
#
## Broadcast this array across 5 columns
#X = a * np.ones((1, 5))
#Y = a * np.ones((1, 2))
#
#
#
#X_train, X_test, Y_train, Y_test = custom_train_test_split(X, Y, window_size, overlap_size, 0.3,  random_state=None)
#
#print("X_train")
#print(X_train)
#print("X_test")
#print(X_test)
count_df = '../genomeCounts_extended_UMI.txt'
tsv_file = '../exon_coverage_input_output/OperonSet.tsv'
tsv_file_2 = '../exon_coverage_input_output/Gene_sequence.tsv'
outpath = '../exon_coverage_input_output/output/'
gene_counts = outpath + 'gene_spacer_counts.csv'
data_window = outpath+'XY_data_Y_with_windows2000_1000.npz'
operon_df = data_loading(tsv_file)
gene_df = data_loading(tsv_file_2)
count_df = pd.read_csv(count_df, sep='\t', dtype=str, low_memory=False)
gene_counts = pd.read_csv(gene_counts, sep=',')
no_bin = 125
binsize = 16
data = np.load(data_window) 


#gene_spacer_counts_normalized_df = get_gene_spacer_counts(count_df, gene_df)
#gene_spacer_counts_normalized_df.to_csv(outpath+'gene_spacer_counts.csv', index=False)

X, Y = data['X'], data['Y']
X = X.astype(np.float32)
Y_normalized_window, Y_normalized, X_norm =  normalize_coverage_per_gene(Y, X, gene_counts, no_bin, binsize)


