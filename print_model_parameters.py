from utils_plotting import plot_predicted_vs_observed
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from custom_elements import  custom_loss_with_l1, poisson_loss, spearman_correlation
from utils_training import filter_annotation_features
##################### Set before plotting #####################
model_path = '/cluster/home/hugifl/spacer_coverage_output_2/3200_1600_gene_norm_outputs/models/CNN_biLSTM_CustomAttention_1'

loaded_model = load_model(model_path, custom_objects={'poisson_loss': poisson_loss}) #, custom_objects={'poisson_loss': poisson_loss} ,, 'spearman_correlation':spearman_correlation
print(loaded_model.summary())