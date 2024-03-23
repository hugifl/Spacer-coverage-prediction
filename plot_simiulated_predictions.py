from utils_plotting import plot_simulated_predictions
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from custom_elements import  custom_loss_with_l1, poisson_loss, spearman_correlation
from utils_simulated_data import filter_annotation_features_simulated
##################### Set before plotting #####################

window_size = 3200
overlap = 1600
no_plots = 1
no_bin = 3200
binsize = 1
model_dataset_name = '3200_1600_gene_norm'
dataset_name = 'dummy_ATCG_two_genes_AG_islets'
dataset_savename = 'dummy_ATCG_two_genes_AG_islets_no_term'
model_to_load = 'CNN_biLSTM_17_3'
model_name = 'CNN_biLSTM_17_3'
coverage_scaling_factor = 1
annotation_features_to_use = ['gene_vector', 'promoter_vector', 'gene_directionality_vector'] # ['gene_vector', 'promoter_vector', 'terminator_vector', 'gene_directionality_vector']
only_seq = False
log_scale = False
print_GC = False
###############################################################
outdir = '../spacer_coverage_output_2/'
datadir = '/cluster/scratch/hugifl/spacer_coverage_final_data_2/Simulated_datasets'
data_file = datadir + '/'+ dataset_name + ".npz"

loaded_model = load_model(outdir + model_dataset_name + "_outputs/models/" + model_to_load, custom_objects={'poisson_loss': poisson_loss}) #, custom_objects={'poisson_loss': poisson_loss} ,, 'spearman_correlation':spearman_correlation
#print(loaded_model.summary())


data = np.load(data_file)
X_test = data['X']



X_test_seq = X_test[:, :, :4]  # Sequence data
X_test_anno = X_test[:, :, 4:] # Annotation data
# Filter the annotation arrays
X_test_anno = filter_annotation_features_simulated(X_test_anno, annotation_features_to_use)

X_test_seq = X_test_seq.astype('float32')
X_test_annot = X_test_anno.astype('float32')


plots = plot_simulated_predictions(loaded_model, model_name, X_test_seq, X_test_anno, no_plots, no_bin, outdir, dataset_savename, binsize, log_scale, annotation_features_to_use, only_seq, print_GC)