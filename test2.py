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
data_dir = '/cluster/scratch/hugifl/spacer_coverage_final_data_2/Transcriptional_Units_TU_norm_V2_data_2/train_test_data_normalized_windows_info_scaled.npz'
data = np.load(data_dir)
X_train = data['X_train']
X_test = data['X_test']
Y_train = data['Y_train']
Y_test = data['Y_test']

print("shape of X_train: ", X_train.shape)
print("shape of Y_train: ", Y_train.shape)

# print row max and min of Y_train for tirst 10 rows
print("max of Y_train for first 10 rows: ", np.max(Y_train[:10,2:], axis=1))

print(Y_train[7,:])
#dir = '/cluster/scratch/hugifl/spacer_coverage_final_data_2/Diet1_Transcriptional_Units_TU_norm_V2_data/TU_coverage_data_summed_TU_removed.csv'
#
#df = pd.read_csv(dir, sep=',', comment="#")
#print("shape of df: ", df.shape)
#print("first 10 columns of df: ", df.columns[:10])
#print("last 10 columns of df: ", df.columns[-10:])
#print("first 7 values of first row of df: ", df.iloc[0, :7])
#print("last 7 values of first row of df: ", df.iloc[0, -7:])
#data_window = '/cluster/scratch/hugifl/spacer_coverage_final_data_2/Diet1_Transcriptional_Units_TU_norm_V2_data/XY_data_Y_with_windows.npz'
#data = np.load(data_window) 
#X, Y = data['X'], data['Y']
#X = X.astype(np.float32)
#
#print("shape of X: ", X.shape)
#print("shape of Y: ", Y.shape)