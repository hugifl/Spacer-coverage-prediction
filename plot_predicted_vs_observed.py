from utils import plot_coverage_predicted_vs_observed_window_info_lines_log, plot_predicted_vs_observed, plot_predicted_vs_observed_window_info, plot_predicted_vs_observed_window_info_lines, plot_coverage_predicted_vs_observed_window_info_lines
import numpy as np
from tensorflow.keras.models import load_model
from custom_layers import MAE_FP_punished_more, sparse_binary_crossentropy, custom_loss_with_l1, poisson_loss
from data_loading import data_loading

window_size = 2000
overlap = 1000

#data = np.load('../exon_coverage_input_output/output/train_test_data_binary_'+str(window_size) + '_' + str(overlap) + '.npz')
data = np.load('../exon_coverage_input_output/output/train_test_data_normalized_windows_info_'+str(window_size) + '_' + str(overlap) + '.npz')

X_train = data['X_train']
X_test = data['X_test']
Y_train = data['Y_train']
Y_test = data['Y_test']

# Find rows with NaNs or Infs in Y_test
#rows_with_nans_or_infs = np.any(np.isnan(Y_test) | np.isinf(Y_test), axis=1)
#Y_test_filtered = Y_test[~rows_with_nans_or_infs]
#X_test_filtered = X_test[~rows_with_nans_or_infs]

# Find indices where the maximum value in a row of Y_test exceeds 30 or is below 2
indices_to_remove_test = np.where((Y_test[:, 2:] > 20).any(axis=1) | (Y_test[:, 2:].max(axis=1) < 2))[0]

# Remove these rows from Y_test and X_test
Y_test_filtered = np.delete(Y_test, indices_to_remove_test, axis=0)
X_test_filtered = np.delete(X_test, indices_to_remove_test, axis=0)
# Remove the channel marking the operon body
#X_test_filtered = X_test_filtered[:, :, :-1]  

small_constant = 1e-6
Y_test_log = np.log10(Y_test_filtered + small_constant)

loaded_model = load_model('../exon_coverage_input_output/output/'+str(window_size) + '_' + str(overlap) + 'CNN_BiLSTM_custom_pooling_2_poisson', custom_objects={'poisson_loss': poisson_loss}) #, custom_objects={'MAE_FP_punished_more': MAE_FP_punished_more}

no_plots = 100
no_bin = 125
binsize = 16
window_size = 2000
tsv_file = '../exon_coverage_input_output/OperonSet.tsv'
tsv_file_2 = '../exon_coverage_input_output/Gene_sequence.tsv'
outpath = '../exon_coverage_input_output/output/prediction_plots/poisson/'

operon_df = data_loading(tsv_file)
gene_df = data_loading(tsv_file_2)
#plots = plot_predicted_vs_observed(X_test, Y_test, loaded_model, no_plots, no_bin, outpath,str(window_size) + '_' + str(overlap) + 'CNN_binary_BiLSTM_custom_pooling_2')
#print(plots)

#plots = plot_coverage_predicted_vs_observed_window_info_lines(X_test_filtered, Y_test_filtered, loaded_model, no_plots, no_bin, outpath, str(window_size) + '_' + str(overlap) + 'CNN_BiLSTM_custom_pooling_2_log', window_size, operon_df, gene_df, binsize)
#print(plots)

plots = plot_coverage_predicted_vs_observed_window_info_lines_log(X_test_filtered, Y_test_filtered, loaded_model, no_plots, no_bin, outpath, str(window_size) + '_' + str(overlap) + 'CNN_BiLSTM_custom_pooling_2_poisson_3', window_size, operon_df, gene_df, binsize)
print(plots)