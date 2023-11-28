from utils_coverage import plot_window_coverage_binarized, plot_window_coverage_normalized
import argparse
from data_loading import data_loading
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()

optional = parser._action_groups.pop()
required = parser.add_argument_group('required arguments')  # change grouping of argument to required or optional in help through this chunk of code


required.add_argument('-op', '--operons', help='genome annotation file path (regulonDB tsv)', required=True)
required.add_argument('-ge', '--genes', help='genome annotation file path (regulonDB tsv)', required=True)
required.add_argument('-pl', '--plots', help='numbers of plots to be generated', required=True)
required.add_argument('-cdf', '--coverage_df', help='coverage dataframe (df)', required = True)
required.add_argument('-cdfn', '--coverage_df_norm', help='normalized coverage dataframe (np.array)', required = True)
optional.add_argument('-o', '--outPath', help='path to output directory.', default='.')
optional.add_argument('-bs', '--binsize', help='binsize to average coverage, power of 2', type=int, default=16)
optional.add_argument('-w', '--winwidth', help='sequence length of input sequence, default = 2000', type=int, dest='winwidth', default=2000)
optional.add_argument('-ov', '--overlap', help='overlap between windows', type=int, dest='overlap', default=1000)

#initialize values

parser._action_groups.append(optional) 
args = parser.parse_args()

tsv_file = str(args.operons)
tsv_file_2 = str(args.genes)
outdir=str(args.outPath)
window_size = int(args.winwidth)
overlap = int(args.overlap)
plots = int(args.plots)
binsize = int(args.binsize)
coverage_df_file = str(args.coverage_df)
norm_coverage_df_file = str(args.coverage_df_norm)

no_bin = window_size / binsize

operon_dataframe = data_loading(tsv_file)
gene_dataframe = data_loading(tsv_file_2)
coverage_df_summed = pd.read_csv(coverage_df_file, sep=',', comment="#")

data = np.load(norm_coverage_df_file)
X_train = data['X_train']
X_test = data['X_test']
Y_train = data['Y_train']
Y_test = data['Y_test']

plots_binarized = plot_window_coverage_binarized(coverage_df_summed, window_size, operon_dataframe, gene_dataframe, plots, no_bin, binsize, outdir, random=False)
print(plots_binarized)
plots_normalized = plot_window_coverage_normalized(Y_test, plots, no_bin, outdir, window_size, operon_dataframe, gene_dataframe, binsize, random=False)
print(plots_normalized)


