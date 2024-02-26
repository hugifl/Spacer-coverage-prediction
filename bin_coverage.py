import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from custom_elements import  poisson_loss, NaNChecker, calculate_pearson_correlation, find_and_plot_peaks, calculate_peak_f1_score, PearsonCorrelationCallback, F1ScoreCallback
from tensorflow.keras.callbacks import EarlyStopping
from utils_training import filter_annotation_features, evaluate_model
from scipy.signal import find_peaks
import os
##################### Set #####################

window_size = 3200
overlap = 1600
no_bin = 800
binsize = 4
dataset_name = '3200_1600_gene_norm'
new_dataset_name = '3200_1600_gene_norm_bin_4'

###############################################################

def bin_coverage_profiles(Y, binsize=4):
    # Split identifiers and coverage data
    identifiers = Y[:, :2]  # First two columns are identifiers
    coverage_data = Y[:, 2:]  # The rest is coverage data
    
    # Determine the new shape
    num_samples, num_columns = coverage_data.shape
    binned_columns = num_columns // binsize
    
    # Reshape and average to bin the coverage data
    binned_coverage = coverage_data.reshape(num_samples, binned_columns, binsize).mean(axis=2)
    
    # Concatenate the identifiers back
    Y_binned = np.concatenate([identifiers, binned_coverage], axis=1)
    
    return Y_binned

data_dir = '/cluster/scratch/hugifl/spacer_coverage_final_data_2/'
data_file_smoothed = data_dir + dataset_name + "_data"+"/train_test_data_normalized_windows_info_smoothed.npz"
data_file = data_dir + dataset_name + "_data"+"/train_test_data_normalized_windows_info.npz"

new_data_dir = '/cluster/scratch/hugifl/spacer_coverage_final_data_2/' + new_dataset_name + "_data/"
if not os.path.exists(new_data_dir):
    os.makedirs(new_data_dir)

data = np.load(data_file)
data_smoothed = np.load(data_file_smoothed)
X_train = data['X_train']
X_test = data['X_test']
Y_train = data['Y_train']
Y_test = data['Y_test']
Y_train_smoothed = data_smoothed['Y_train']
Y_test_smoothed = data_smoothed['Y_test']

Y_train_binned = bin_coverage_profiles(Y_train, binsize=binsize)
Y_test_binned = bin_coverage_profiles(Y_test, binsize=binsize)
Y_train_smoothed_binned = bin_coverage_profiles(Y_train_smoothed, binsize=binsize)
Y_test_smoothed_binned = bin_coverage_profiles(Y_test_smoothed, binsize=binsize)

np.savez(new_data_dir + "train_test_data_normalized_windows_info.npz", X_train=X_train, X_test=X_test, Y_train=Y_train_binned, Y_test=Y_test_binned)
np.savez(new_data_dir + "train_test_data_normalized_windows_info_smoothed.npz", X_train=X_train, X_test=X_test, Y_train=Y_train_smoothed_binned, Y_test=Y_test_smoothed_binned)