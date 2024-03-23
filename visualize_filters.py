import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from custom_elements import  poisson_loss, spearman_correlation
import logomaker
import os
from tensorflow.keras.models import load_model

# Load your trained model
outdir = '/cluster/home/hugifl/spacer_coverage_output_2/filters/'
outpath = '/cluster/home/hugifl/spacer_coverage_output_2/'
model_path = '/cluster/home/hugifl/spacer_coverage_output_2/3200_1600_gene_norm_outputs/models/'
model_name = 'CNN_biLSTM_17'
model_dataset_name = '3200_1600_gene_norm'

if not os.path.exists(outdir + model_name + 'filters/'):
    os.makedirs(outdir + model_name + 'filters/')

# Extract filters from the convolutional layer
loaded_model = load_model(outpath + model_dataset_name + "_outputs/models/" + model_name, custom_objects={'poisson_loss': poisson_loss}) #, custom_objects={'poisson_loss': poisson_loss} ,, 'spearman_correlation':spearman_correlation



conv_layer = loaded_model.get_layer('conv1d')  # Adjust the layer name
filters, biases = conv_layer.get_weights()

#print some example filters
print("shape of filters: ", filters.shape)
print(filters[:,:,0])

average_weights = np.mean(filters, axis=(0, 2))

# Assuming the order of nucleotides is ['A', 'G', 'C', 'T']
nucleotides = ['A', 'G', 'C', 'T']

# Print average weights for each nucleotide
for nucleotide, weight in zip(nucleotides, average_weights):
    print(f"Average weight for {nucleotide}: {weight}")

#print(filters[:,:,2])
## Normalize the filters
#f_min, f_max = filters.min(), filters.max()
#print(f_min, f_max)
#filters = (filters - f_min) / (f_max - f_min)
#print(filters[:,:,2])
## Nucleotide labels
#nucleotides = ['A', 'G', 'C', 'T']
#
## Create sequence logos for each filter
#for i in range(filters.shape[2]):
#    # Prepare the data for logomaker
#    filter_data = filters[:, :, i]
#    keep = False
#    for row in filter_data:
#        if np.max(row) > (row.sum() - 1.5*np.max(row)):   #if np.max(row) - np.min(row) > 0.5
#            keep = True
#    if not keep:
#        continue
#    print(filter_data)
#    filter_df = pd.DataFrame(filter_data, columns=nucleotides)
#
#    # Create the sequence logo
#    logo = logomaker.Logo(filter_df, figsize=(6, 2), color_scheme='classic')
#    logo.style_spines(visible=False)
#    logo.style_spines(spines=['left', 'bottom'], visible=True)
#    logo.ax.set_ylabel("Weight")
#    logo.ax.set_title(f"Filter {i}")
#    plt.savefig(outdir + model_name + 'filters/' + f"anno_filter_logo{i}.png" )
## Normalize filter values to 0-1 so we can visualize them
#f_min, f_max = filters.min(), filters.max()
#filters = (filters - f_min) / (f_max - f_min)
#
## Plot first few filters
#n_filters = min(9, filters.shape[2])  # Number of filters to display, up to 6
#n_columns = 3  # Number of columns in display grid
#
#fig, axes = plt.subplots(int(n_filters / n_columns), n_columns, figsize=(n_columns*2, int(n_filters / n_columns)*2))
#axes = axes.flatten()
#
#for i in range(n_filters):
#    # Get the filter
#    f = filters[:, :, i]
#
#    # Plot the filter coefficients
#    ax = axes[i]
#    ax.plot(f)
#    ax.set_title(f'Filter {i}')
#    ax.axis('off')
#    plt.savefig(outdir + model_name + f'filters/filter_weight{i}.png')