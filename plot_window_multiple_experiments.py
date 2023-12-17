from utils_plotting import plot_window_coverage_normalized_compare_profiles
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()

optional = parser._action_groups.pop()
required = parser.add_argument_group('required arguments')  # change grouping of argument to required or optional in help through this chunk of code

required.add_argument('-da1', '--dataset_name_1', help='name of the first dataset at use', required=True)
required.add_argument('-dan1', '--dataset_name_plot_1', help='name of the first dataset at use', required=True)
required.add_argument('-da2', '--dataset_name_2', help='name of the second dataset at use', required=True)
required.add_argument('-dan2', '--dataset_name_plot_2', help='name of the second dataset at use', required=True)
required.add_argument('-pr', '--promoters', help='genome annotation file path (ECOCYC tsv)', required=True)
required.add_argument('-te', '--terminators', help='genome annotation file path (ECOCYC tsv)', required=True)
required.add_argument('-ge', '--genes', help='genome annotation file path (regulonDB tsv)', required=True)
required.add_argument('-pl', '--plots', help='numbers of plots to be generated', required=True)
required.add_argument('-cdfn', '--data', help='normalized coverage dataframe (np.array)', required = True)
optional.add_argument('-o', '--outPath', help='path to output directory.', default='.')
optional.add_argument('-d', '--data_dir', help='path to data direcotry', default='.')
optional.add_argument('-bs', '--binsize', help='binsize to average coverage, power of 2', type=int, default=16)
optional.add_argument('-w', '--winwidth', help='sequence length of input sequence, default = 2000', type=int, dest='winwidth', default=2000)
optional.add_argument('-ov', '--overlap', help='overlap between windows', type=int, dest='overlap', default=1000)

#initialize values

parser._action_groups.append(optional) 
args = parser.parse_args()

dataset_1 = str(args.dataset_name_1)
dataset_2 = str(args.dataset_name_2)
dataset_name_1 = str(args.dataset_name_plot_1)
dataset_name_2 = str(args.dataset_name_plot_2)
promoter_file = str(args.promoters)
terminator_file = str(args.terminators)
gene_file = str(args.genes)
outdir=str(args.outPath)
data_dir=str(args.data_dir)
window_size = int(args.winwidth)
overlap = int(args.overlap)
plots = int(args.plots)
binsize = int(args.binsize)
data_file = str(args.data)
data_file_1 = data_dir + dataset_1 + "_data"+"/"+ data_file
data_file_2 = data_dir + dataset_2 + "_data"+"/"+ data_file
no_bin = window_size / binsize

promoter_df = pd.read_csv(promoter_file, sep='\t')
promoter_df.dropna(inplace=True)

terminator_df = pd.read_csv(terminator_file, sep='\t')
terminator_df.dropna(inplace=True)


gene_df = pd.read_csv(gene_file, sep='\t')
gene_df.drop(gene_df.columns[1], axis=1, inplace=True)
gene_df.dropna(inplace=True)


data_1 = np.load(data_file_1)
X_train_1 = data_1['X_train']
X_test_1 = data_1['X_test']
Y_train_1 = data_1['Y_train']
Y_test_1 = data_1['Y_test']

data_2 = np.load(data_file_2)
X_train_2 = data_2['X_train']
X_test_2 = data_2['X_test']
Y_train_2 = data_2['Y_train']
Y_test_2 = data_2['Y_test']


#plots_binarized = plot_window_coverage_binarized(coverage_df_summed, window_size, operon_dataframe, gene_dataframe, plots, no_bin, binsize, outdir, random=False)
#print(plots_binarized)
plots_normalized = plot_window_coverage_normalized_compare_profiles(Y_train_1, Y_train_2,dataset_name_1,dataset_name_2,plots, no_bin, outdir, window_size,promoter_df, terminator_df, gene_df, binsize, random=False)
print(plots_normalized)