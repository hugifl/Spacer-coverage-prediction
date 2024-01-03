import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from custom_elements import  poisson_loss, spearman_correlation
import logomaker
import os

# Load your trained model
outdir = '/cluster/home/hugifl/spacer_coverage_output/filters/'
model_path = '/cluster/home/hugifl/spacer_coverage_output/window_3200_overlapt_1600_binsize_4_outputs/models/'
model_name = 'CNN_BiLSTM_custom_pooling_dual_input_4_3'

if not os.path.exists(outdir + model_name + 'filters/'):
    os.makedirs(outdir + model_name + 'filters/')
# Extract filters from the convolutional layer
model =  tf.keras.models.load_model(model_path + model_name , custom_objects={'poisson_loss': poisson_loss, 'spearman_correlation':spearman_correlation})

conv_layer = model.get_layer('conv1d_2')  # Adjust the layer name
filters, biases = conv_layer.get_weights()

print(filters[:,:,2])
# Normalize the filters
f_min, f_max = filters.min(), filters.max()
print(f_min, f_max)
filters = (filters - f_min) / (f_max - f_min)
print(filters[:,:,2])
# Nucleotide labels
nucleotides = ['A', 'G', 'C', 'T']

# Create sequence logos for each filter
for i in range(filters.shape[2]):
    # Prepare the data for logomaker
    filter_data = filters[:, :, i]
    keep = False
    for row in filter_data:
        if np.max(row) > (row.sum() - 1.5*np.max(row)):   #if np.max(row) - np.min(row) > 0.5
            keep = True
    if not keep:
        continue
    print(filter_data)
    filter_df = pd.DataFrame(filter_data, columns=nucleotides)

    # Create the sequence logo
    logo = logomaker.Logo(filter_df, figsize=(6, 2), color_scheme='classic')
    logo.style_spines(visible=False)
    logo.style_spines(spines=['left', 'bottom'], visible=True)
    logo.ax.set_ylabel("Weight")
    logo.ax.set_title(f"Filter {i}")
    plt.savefig(outdir + model_name + 'filters/' + f"anno_filter_logo{i}.png" )

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
#    plt.savefig(f"/cluster/home/hugifl/spacer_coverage_output/test/filter_weight{i}.png")