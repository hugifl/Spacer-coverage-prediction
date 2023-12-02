import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils_coverage import lowest_expressed_genes, filter_windows, get_windows, define_peaks, custom_train_test_split
from utils_coverage import normalize_coverage_per_operon, normalize_coverage_per_gene
import argparse

parser = argparse.ArgumentParser()

optional = parser._action_groups.pop()
required = parser.add_argument_group('required arguments')  # change grouping of argument to required or optional in help through this chunk of code

### inputs ###
# required
required.add_argument('-opc', '--counts', help='file with spacer counts per normalization unit', required=True)
required.add_argument('-nu', '--normalization_unit', help='operon or gene to normalize coverage profiles per operon or per gene', required=True)

# optional
optional.add_argument('-o', '--outPath', help='path to output directory.', default='.')
optional.add_argument('-w', '--winwidth', help='sequence length of input sequence, default = 1000', type=int, dest='winwidth', default=2000)
optional.add_argument('-ov', '--overlap', help='overlap between windows', type=int, dest='overlap', default=400)
optional.add_argument('-bs', '--binsize', help='binsize', type=int, default=16)

#initialize values

parser._action_groups.append(optional) 
args = parser.parse_args()

count_file = str(args.counts)
normalization_unit = str(args.normalization_unit)
outdir=str(args.outPath)
window_size = int(args.winwidth)
binsize = int(args.binsize)
overlap_size = int(args.overlap)

no_bin = int(window_size/binsize)
#data_no_window = outdir+'XY_data_'+str(window_size)+'_'+str(overlap_size) +'.npz'
data_window = outdir+'XY_data_Y_with_windows'+str(window_size)+'_'+str(overlap_size) +'.npz'
spacer_counts_df = pd.read_csv(count_file, sep=',', comment="#")


if normalization_unit == 'operon':
    data = np.load(data_window) 
    X, Y = data['X'], data['Y']
    X = X.astype(np.float32)
    Y_normalized_window, Y_normalized, X_norm =  normalize_coverage_per_operon(Y, X, spacer_counts_df, no_bin, binsize)
    #Y_binary_window = define_peaks(Y_normalized_window, 0.3, 0.0000001, window_info="Yes")

elif normalization_unit == 'gene':
    data = np.load(data_window) 
    X, Y = data['X'], data['Y']
    X = X.astype(np.float32)
    Y_normalized_window, Y_normalized, X_norm =  normalize_coverage_per_gene(Y, X, spacer_counts_df, no_bin, binsize)
   # Y_binary_window = define_peaks(Y_normalized_window, 0.3, 0.001, window_info="Yes")


X_train, X_test, Y_train, Y_test = custom_train_test_split(X_norm, Y_normalized_window, window_size, overlap_size, 0.05,  random_state=None)
np.savez(outdir + 'train_test_data_normalized_windows_info_' + str(window_size)+'_'+str(overlap_size) + '.npz', X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test)
#X_train, X_test, Y_train, Y_test = custom_train_test_split(X_norm, Y_binary_window, window_size, overlap_size, 0.05,  random_state=None)
#np.savez(outdir + 'train_test_data_binary_windows_info_' + str(window_size)+'_'+str(overlap_size) + '.npz', X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test)
