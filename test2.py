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
dataset_name = '3200_1600_gene_norm_bin_4'

###############################################################

data_dir = '/cluster/scratch/hugifl/spacer_coverage_final_data_2/'
data_file_smoothed = data_dir + dataset_name + "_data"+"/train_test_data_normalized_windows_info_smoothed.npz"
data_file = data_dir + dataset_name + "_data"+"/train_test_data_normalized_windows_info.npz"


data = np.load(data_file)
data_smoothed = np.load(data_file_smoothed)
X_train = data['X_train']
X_test = data['X_test']
Y_train = data['Y_train']
Y_test = data['Y_test']

Y_train_smoothed = data_smoothed['Y_train']
Y_test_smoothed = data_smoothed['Y_test']

print("Y_train_smoothed.shape: ", Y_train_smoothed.shape)
print("Y_test_smoothed.shape: ", Y_test_smoothed.shape)
print("Y_train.shape: ", Y_train.shape)
print("Y_test.shape: ", Y_test.shape)