from utils_plotting import plot_window_coverage_normalized_TU
import argparse
import pandas as pd
import numpy as np

no_plots = 40
binsize = 1
dataset_name = 'Transcriptional_Units_TU_norm_V2'
annotation_features_to_use = ['gene_vector'] # ['gene_vector', 'promoter_vector', 'terminator_vector', 'gene_directionality_vector']
###############################################################
outdir = '../spacer_coverage_output_2/'
datadir = '/cluster/scratch/hugifl/spacer_coverage_final_data_2/'
data_file = datadir + dataset_name + "_data"+"/train_test_data_normalized_windows_info.npz"

promoter_file = '../spacer_coverage_input/ECOCYC_promoters.txt'
terminator_file = '../spacer_coverage_input/ECOCYC_terminators.txt'
gene_file = '../spacer_coverage_input/ECOCYC_genes.txt'

promoter_df = pd.read_csv(promoter_file, sep='\t')
promoter_df.dropna(inplace=True)

terminator_df = pd.read_csv(terminator_file, sep='\t')
terminator_df.dropna(inplace=True)

gene_df = pd.read_csv(gene_file, sep='\t')
gene_df.drop(gene_df.columns[1], axis=1, inplace=True)
gene_df.dropna(inplace=True)


data = np.load(data_file)
X_train = data['X_train']
X_test = data['X_test']
Y_train = data['Y_train']
Y_test = data['Y_test']

scaling_factor = 1

Y_test[:,2:] = Y_test[:,2:] 
Y_train[:,2:] = Y_train[:,2:] 

#plots_binarized = plot_window_coverage_binarized(coverage_df_summed, window_size, operon_dataframe, gene_dataframe, plots, no_bin, binsize, outdir, random=False)
#print(plots_binarized)
plots_normalized = plot_window_coverage_normalized_TU(Y_test, no_plots, outdir, dataset_name, promoter_df, terminator_df, gene_df, binsize,random = False)
print(plots_normalized)
