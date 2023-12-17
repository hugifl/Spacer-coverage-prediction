import numpy as np
import pandas as pd
import os
from utils_train_test_data import custom_train_test_split, normalize_coverage_per_gene, replace_ones, gaussian_smooth_profiles

##################### Dataset information #####################
# Make sure to only combine datasets with the same window and binsize
datasets = ['window_3200_overlapt_1600_binsize_4', 'paraquat_window_3200_overlapt_1600_binsize_4']
combined_dataset_name = "In_vivo_gut_In_vitro_paraquat"
window_size = 3200
overlap_size = 1600
###############################################################

data_dir = '/cluster/scratch/hugifl/spacer_coverage_final_data/'
outdir = data_dir+ "_data/" + combined_dataset_name + "/"

if not os.path.exists(outdir):
    os.makedirs(outdir)

total_library_size = 0
library_sizes = []

# Calculate total library size and store individual sizes
for dataset_name in datasets:
    library_size_file = data_dir + dataset_name + "_data/" + 'tot_number_aligned_reads.txt'
    with open(library_size_file, 'r') as file:
        library_size = int(file.read().strip())
        total_library_size += library_size
        library_sizes.append(library_size)

# Initialize empty array for combined coverage
combined_coverage = None

# Process each dataset
for index, dataset_name in enumerate(datasets):
    data_file = data_dir + dataset_name + "_data/" + "XY_data_Y_with_windows.npz"
    data = np.load(data_file)
    X = data['X']
    X = X.astype(np.float32)
    Y = data['Y']
    
    # Calculate weight for this dataset
    weight = library_sizes[index] / total_library_size
    
    # Weight the coverage data
    weighted_coverage = Y[:, 2:] * weight

    # Add to combined coverage
    if combined_coverage is None:
        combined_coverage = np.zeros_like(weighted_coverage)
    combined_coverage += weighted_coverage

# Combined coverage now contains the summed coverage, weighted by library size
# The window information (first two columns) can be added back if necessary
final_Y = np.hstack((X[:, :2], combined_coverage))

final_Y_smoothed = gaussian_smooth_profiles(final_Y, sigma=3)
np.savez(outdir+'XY_data_Y_with_windows_smoothed.npz', X=X, Y=final_Y_smoothed)


X_train, X_test, Y_train, Y_test = custom_train_test_split(X, final_Y_smoothed, window_size, overlap_size, 0.05,  random_state=None)
np.savez(outdir + 'train_test_data_normalized_windows_info_smoothed.npz', X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test)
