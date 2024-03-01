import numpy as np
from utils_training import filter_annotation_features
import os
import pandas as pd
##################### Set #####################

window_size = 3200
overlap = 1600
no_bin = 800
binsize = 4
dataset_name = '3200_1600_gene_norm'

###############################################################
outdir = '../spacer_coverage_output_2/'
data_dir = '/cluster/scratch/hugifl/spacer_coverage_final_data_2/'
data_file = data_dir + dataset_name + "_data"+"/train_test_data_normalized_windows_info.npz"

data = np.load(data_file)
X_train = data['X_train']
X_test = data['X_test']
Y_train = data['Y_train']
Y_test = data['Y_test']

# Adjust the coverage data
Y_train = Y_train[:, 2:]
Y_test = Y_test[:, 2:]



# Find rows with NaNs or Infs in Y_train
rows_with_nans_or_infs = np.any(np.isnan(Y_train) | np.isinf(Y_train), axis=1)
Y_train_filtered = Y_train[~rows_with_nans_or_infs]
X_train_filtered = X_train[~rows_with_nans_or_infs]

# Find rows with NaNs or Infs in Y_test
rows_with_nans_or_infs = np.any(np.isnan(Y_test) | np.isinf(Y_test), axis=1)
Y_test_filtered = Y_test[~rows_with_nans_or_infs]
X_test_filtered = X_test[~rows_with_nans_or_infs]


# Filter out windows that contain genes with coverage peaks too high (normalization error due to wrong/non-matching coordinates) or too low (low gene expression, noisy profile)
#indices_to_remove_train = np.where((Y_train_filtered > 200).any(axis=1) | (Y_train_filtered.max(axis=1) < 5))[0]
##
### Remove these rows from Y_train and X_train
#Y_train_filtered = np.delete(Y_train_filtered, indices_to_remove_train, axis=0)
#X_train_filtered = np.delete(X_train_filtered, indices_to_remove_train, axis=0)
#
## Find indices where the maximum value in a row of Y_test exceeds 20 or is below 2
#indices_to_remove_test = np.where((Y_test_filtered > 200).any(axis=1) | (Y_test_filtered.max(axis=1) < 5))[0]
##
### Remove these rows from Y_test and X_test
#Y_test_filtered = np.delete(Y_test_filtered, indices_to_remove_test, axis=0)
#X_test_filtered = np.delete(X_test_filtered, indices_to_remove_test, axis=0)

#Y_train_binarized = (Y_train_filtered > 2).astype(int)
#Y_test_binarized = (Y_test_filtered > 2).astype(int)


    



# Adjust the input data
X_train_seq = X_train_filtered[:, :, :4]  # Sequence data
X_train_anno = X_train_filtered[:, :, 4:] # Annotation data

X_test_seq = X_test_filtered[:, :, :4]  # Sequence data
X_test_anno = X_test_filtered[:, :, 4:] # Annotation data

print("dimension of X_train_seq: ", X_train_seq.shape)
print("dimension of X_train_anno: ", X_train_anno.shape)


# Filter the annotation arrays
X_train_anno, X_test_anno = filter_annotation_features(X_train_anno, X_test_anno, ['gene_vector', 'promoter_vector', 'gene_directionality_vector'])

print("shape of X_train_anno: ", X_train_anno.shape)
print("shape of X_test_anno: ", X_test_anno.shape)