from utils_coverage import plot_window_coverage
import argparse
from data_loading import data_loading
import pandas as pd

parser = argparse.ArgumentParser()

optional = parser._action_groups.pop()
required = parser.add_argument_group('required arguments')  # change grouping of argument to required or optional in help through this chunk of code


required.add_argument('-op', '--operons', help='genome annotation file path (regulonDB tsv)', required=True)
required.add_argument('-ge', '--genes', help='genome annotation file path (regulonDB tsv)', required=True)
required.add_argument('-pl', '--plots', help='numbers of plots to be generated', required=True)
required.add_argument('-cdf', '--coverage_df', help='coverage dataframe', required = True)
optional.add_argument('-o', '--outPath', help='path to output directory.', default='.')
optional.add_argument('-bs', '--binsize', help='binsize to average coverage, power of 2', type=int, default=16)
optional.add_argument('-w', '--winwidth', help='sequence length of input sequence, default = 2000', type=int, dest='winwidth', default=2000)

#initialize values

parser._action_groups.append(optional) 
args = parser.parse_args()

tsv_file = str(args.operons)
tsv_file_2 = str(args.genes)
outdir=str(args.outPath)
window_size = int(args.winwidth)
plots = int(args.plots)
binsize = int(args.binsize)
coverage_df_file = str(args.coverage_df)

no_bin = window_size / binsize

operon_dataframe = data_loading(tsv_file)
gene_dataframe = data_loading(tsv_file_2)
coverage_df_summed = pd.read_csv(coverage_df_file, sep=',', comment="#")

plots = plot_window_coverage(coverage_df_summed, window_size, operon_dataframe, gene_dataframe, plots, no_bin, binsize, outdir)
print(plots)
