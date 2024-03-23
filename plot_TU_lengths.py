#from models import CNN_BiLSTM_custom_pooling_dual_input_4_3, CNN_BiLSTM_custom_pooling_dual_input_4_2, CNN_BiLSTM_custom_pooling_dual_input_4 ,CNN_BiLSTM_custom_pooling_dual_input, CNN_BiLSTM_custom_pooling_dual_input_2, CNN_BiLSTM_avg_pooling_4_dual_input, CNN_BiLSTM_avg_pooling_4_dual_input_2
from models_2 import CNN_biLSTM_1_Masking
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint
from utils_training import filter_annotation_features_TU, evaluate_model, custom_batch_generator, clip_test_set, calculate_total_batches, restrict_TU_lengths
from scipy.signal import find_peaks
import os
import re
import glob
import pandas as pd
from utils_plotting import plot_predicted_vs_observed_TU_during_training
##################### Set before training #####################

binsize = 1
batch_size = 64
dataset_name = 'Transcriptional_Units_TU_norm_V2'
pad_symbol = 0.420

###############################################################
outdir = '../spacer_coverage_output_2/'
data_dir = '/cluster/scratch/hugifl/spacer_coverage_final_data_2/'
data_file = data_dir + dataset_name + "_data"+"/train_test_data_normalized_windows_info.npz"

data = np.load(data_file)
X_train = data['X_train']
X_test = data['X_test']
Y_train = data['Y_train']
Y_test = data['Y_test']

##############################################################


# Find rows with NaNs or Infs in Y_train
rows_with_nans_or_infs = np.any(np.isnan(Y_train) | np.isinf(Y_train), axis=1)
Y_train_filtered = Y_train[~rows_with_nans_or_infs]
X_train_filtered = X_train[~rows_with_nans_or_infs]

# Find rows with NaNs or Infs in Y_test
rows_with_nans_or_infs = np.any(np.isnan(Y_test) | np.isinf(Y_test), axis=1)
Y_test_filtered = Y_test[~rows_with_nans_or_infs]
X_test_filtered = X_test[~rows_with_nans_or_infs]

X_test_filtered, Y_test_filtered = restrict_TU_lengths(X_test_filtered, Y_test_filtered, min_length=200, max_length=3000)
X_train_filtered, Y_train_filtered = restrict_TU_lengths(X_train_filtered, Y_train_filtered, min_length=200, max_length=3000)


train_TU_lengths = Y_train_filtered[:,1] - Y_train_filtered[:,0]
test_TU_lengths = Y_test_filtered[:,1] - Y_test_filtered[:,0]


# Adjust the input data
X_train_seq = X_train_filtered[:, :, :4]  # Sequence data
X_train_anno = X_train_filtered[:, :, 4:] # Annotation data

X_test_seq = X_test_filtered[:, :, :4]  # Sequence data
X_test_anno = X_test_filtered[:, :, 4:] # Annotation data

X_test_seq_eval, X_test_anno_eval, Y_test_filtered_eval = clip_test_set(X_test_seq, X_test_anno, Y_test_filtered)    


train_generator = custom_batch_generator(X_train_seq, X_train_anno, Y_train_filtered, batch_size)
validation_generator = custom_batch_generator(X_test_seq, X_test_anno, Y_test_filtered, batch_size)

plot_direcory = outdir + dataset_name + "_outputs/"



plt.figure(figsize=(10, 6))
sns.histplot(train_TU_lengths, bins=100, kde=True, stat="density", linewidth=0, color='blue', label='Train Set', alpha=0.7)
sns.histplot(test_TU_lengths, bins=100, kde=True, stat="density", linewidth=0, color='red', label='Test Set', alpha=0.7)
plt.title('Density Distribution of Transcriptional Unit (TU) Lengths in Train and Test Sets')
plt.xlabel('TU Length')
plt.ylabel('Density')
plt.legend()
plt.grid(axis='y', alpha=0.75)
plt.xticks(ticks=range(0, int(max(train_TU_lengths)), 500), rotation=45)
plt.savefig(plot_direcory + "TU_length_distribution_density_restr.png")

plt.figure(figsize=(10, 6))
sns.histplot(train_TU_lengths, bins=100, kde=True, stat="density", linewidth=0, color='blue', label='Train Set', alpha=0.7)
sns.histplot(test_TU_lengths, bins=100, kde=True, stat="density", linewidth=0, color='red', label='Test Set', alpha=0.7)
plt.title('Density Distribution of Transcriptional Unit (TU) Lengths in Train and Test Sets')
plt.xlabel('TU Length')
plt.ylabel('Density')
plt.legend()
plt.xlim(0, 500)
plt.grid(axis='y', alpha=0.75)
plt.xticks(ticks=range(0, 1000, 25), rotation=45)
plt.savefig(plot_direcory + "TU_length_distribution_density_short_TUs_restr.png")
