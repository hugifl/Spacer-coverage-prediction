from utils_plotting import plot_predicted_vs_observed
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from custom_elements import  custom_loss_with_l1, poisson_loss, spearman_correlation
from utils_training import filter_annotation_features
##################### Set before plotting #####################

window_size = 3200
overlap = 1600
no_plots = 40
no_bin = 3200
binsize = 1
dataset_name = '3200_1600_gene_norm'
model_to_load = 'CNN_biLSTM_10'
model_name = 'CNN_biLSTM_10'
coverage_scaling_factor = 1
annotation_features_to_use = ['gene_vector', 'promoter_vector', 'terminator_vector', 'gene_directionality_vector']
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


loaded_model = load_model(outdir + dataset_name + "_outputs/models/" + model_to_load, custom_objects={'poisson_loss': poisson_loss}) #, custom_objects={'poisson_loss': poisson_loss} ,, 'spearman_correlation':spearman_correlation
#print(loaded_model.summary())

data = np.load(data_file)
X_train = data['X_train']
X_test = data['X_test']
Y_train = data['Y_train']
Y_test = data['Y_test']
# Adjust the coverage data



Y_test[:,2:] = Y_test[:,2:] * coverage_scaling_factor
Y_train[:,2:] = Y_train[:,2:] * coverage_scaling_factor


# Find rows with NaNs or Infs in Y_train
rows_with_nans_or_infs = np.any(np.isnan(Y_train) | np.isinf(Y_train), axis=1)
Y_train_filtered = Y_train[~rows_with_nans_or_infs]
X_train_filtered = X_train[~rows_with_nans_or_infs]

# Find rows with NaNs or Infs in Y_test
rows_with_nans_or_infs = np.any(np.isnan(Y_test) | np.isinf(Y_test), axis=1)
Y_test_filtered = Y_test[~rows_with_nans_or_infs]
X_test_filtered = X_test[~rows_with_nans_or_infs]

# Filter out windows that contain genes with coverage peaks too high (normalization error due to wrong/non-matching coordinates) or too low (low gene expression, noisy profile)
#indices_to_remove_train = np.where((Y_train_filtered[:,2:] > 200).any(axis=1) | (Y_train_filtered[:,2:].max(axis=1) < 5))[0]
##
### Remove these rows from Y_train and X_train
#Y_train_filtered = np.delete(Y_train_filtered, indices_to_remove_train, axis=0)
#X_train_filtered = np.delete(X_train_filtered, indices_to_remove_train, axis=0)
#
### Find indices where the maximum value in a row of Y_test exceeds 20 or is below 2
#indices_to_remove_test = np.where((Y_test_filtered[:,2:] > 200).any(axis=1) | (Y_test_filtered[:,2:].max(axis=1) < 5))[0]
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

# Filter the annotation arrays
X_train_anno, X_test_anno = filter_annotation_features(X_train_anno, X_test_anno, annotation_features_to_use)

print("head of Y_test_filtered", Y_test_filtered[:10, :5])

#plots = plot_predicted_vs_observed(loaded_model, model_name, X_test_seq, X_test_anno, Y_test_filtered, no_plots, no_bin, outdir, dataset_name, window_size, promoter_df, terminator_df, gene_df, binsize, log_scale = False)
#print(plots)