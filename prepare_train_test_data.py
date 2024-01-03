import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils_train_test_data import custom_train_test_split, normalize_coverage_per_gene, replace_ones, gaussian_smooth_profiles
import argparse

parser = argparse.ArgumentParser()

optional = parser._action_groups.pop()
required = parser.add_argument_group('required arguments')  # change grouping of argument to required or optional in help through this chunk of code

### inputs ###
# required
required.add_argument('-opc', '--counts', help='file with spacer counts per gene unit', required=True)

# optional
optional.add_argument('-o', '--outPath', help='path to output directory.', default='.')
optional.add_argument('-w', '--winwidth', help='sequence length of input sequence, default = 1000', type=int, dest='winwidth', default=2000)
optional.add_argument('-ov', '--overlap', help='overlap between windows', type=int, dest='overlap', default=400)
optional.add_argument('-bs', '--binsize', help='binsize', type=int, default=16)

#initialize values

parser._action_groups.append(optional) 
args = parser.parse_args()

count_file = str(args.counts)
outdir=str(args.outPath)
window_size = int(args.winwidth)
binsize = int(args.binsize)
overlap_size = int(args.overlap)

no_bin = int(window_size/binsize)
data_window = outdir+'XY_data_Y_with_windows.npz'
spacer_counts_df = pd.read_csv(count_file, sep=',', comment="#")



data = np.load(data_window) 
X, Y = data['X'], data['Y']
X = X.astype(np.float32)

X_train, X_test, Y_train, Y_test = custom_train_test_split(X, Y, window_size, overlap_size, 0.05,  random_state=None)
np.savez(outdir + 'train_test_data_normalized_windows_info_.npz', X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test)

Y_test_smoothed = gaussian_smooth_profiles(Y_test, sigma=3)
Y_train_smoothed = gaussian_smooth_profiles(Y_train, sigma=3)

np.savez(outdir + 'train_test_data_normalized_windows_info.npz', X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test)
np.savez(outdir + 'train_test_data_normalized_windows_info_smoothed.npz', X_train=X_train, X_test=X_test, Y_train=Y_train_smoothed, Y_test=Y_test_smoothed)
