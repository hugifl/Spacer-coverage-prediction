from utils import plot_coverage_predicted_vs_observed_window_info_lines_log, plot_coverage_predicted_vs_observed_window_info_lines_binary, plot_predicted_vs_observed, plot_predicted_vs_observed_window_info, plot_predicted_vs_observed_window_info_lines
import numpy as np
from tensorflow.keras.models import load_model
from custom_elements import  custom_loss_with_l1, poisson_loss
from data_loading import data_loading

window_size = 3200
overlap = 1602
no_plots = 70
no_bin = 1600
binsize = 2

#data = np.load('../exon_coverage_input_output/output/train_test_data_binary_'+str(window_size) + '_' + str(overlap) + '.npz')
data = np.load('../exon_coverage_input_output/output/train_test_data_normalized_windows_info_'+str(window_size) + '_' + str(overlap) + '.npz')

X_train = data['X_train']
X_test = data['X_test']
Y_train = data['Y_train']
Y_test = data['Y_test']


scaling_factor = 1e-6

Y_test[:,2:] = Y_test[:,2:]  * scaling_factor
Y_train = Y_train * scaling_factor
# Find rows with NaNs or Infs in Y_test
#rows_with_nans_or_infs = np.any(np.isnan(Y_test) | np.isinf(Y_test), axis=1)
#Y_test_filtered = Y_test[~rows_with_nans_or_infs]
#X_test_filtered = X_test[~rows_with_nans_or_infs]

# Find indices where the maximum value in a row of Y_test exceeds 30 or is below 2
indices_to_remove_test = np.where((Y_test[:, 2:] > 15).any(axis=1) | (Y_test[:, 2:].max(axis=1) < 2))[0]

# Remove these rows from Y_test and X_test
Y_test_filtered = np.delete(Y_test, indices_to_remove_test, axis=0)
X_test_filtered = np.delete(X_test, indices_to_remove_test, axis=0)

# Binarize coverage profiles
Y_test_binarized_third_onwards = (Y_test_filtered[:, 2:] > 2).astype(int)
Y_test_binarized = np.concatenate([Y_test_filtered[:, :2], Y_test_binarized_third_onwards], axis=1)

## Remove the channel marking the operon body
##X_test_filtered = X_test_filtered[:, :, :-1]  

#small_constant = 1e-6
#Y_test_log = np.log10(Y_test_filtered + small_constant)

loaded_model = load_model('../exon_coverage_input_output/output/models_'+str(window_size) + '_' + str(overlap) + '_bin_2_unnormalized' +'/' +str(window_size) + '_' + str(overlap)  + 'CNN_BiLSTM_custom_pooling_poisson_bin2_3', custom_objects={'poisson_loss': poisson_loss}) #, custom_objects={'poisson_loss': poisson_loss}


tsv_file = '../exon_coverage_input_output/OperonSet.tsv'
tsv_file_2 = '../exon_coverage_input_output/Gene_sequence.tsv'
outpath = '../exon_coverage_input_output/output/prediction_plots/plots_' + str(window_size) + '_' + str(overlap)+ '_bin_2_unnormalized'+'/'+ 'poisson/'

operon_df = data_loading(tsv_file)
gene_df = data_loading(tsv_file_2)
#plots = plot_predicted_vs_observed(X_test, Y_test, loaded_model, no_plots, no_bin, outpath,str(window_size) + '_' + str(overlap) + 'CNN_binary_BiLSTM_custom_pooling_2')
#print(plots)

#plots = plot_coverage_predicted_vs_observed_window_info_lines_binary(X_test_filtered, Y_test_binarized, loaded_model, no_plots, no_bin, outpath, str(window_size) + '_' + str(overlap) + 'CNN_BiLSTM_custom_pooling_binary', window_size, operon_df, gene_df, binsize)
#print(plots)

plots = plot_coverage_predicted_vs_observed_window_info_lines_log(X_test_filtered, Y_test_filtered, loaded_model, no_plots, no_bin, outpath, str(window_size) + '_' + str(overlap) + 'CNN_BiLSTM_custom_pooling_poisson_bin2_3', window_size, operon_df, gene_df, binsize)
print(plots)