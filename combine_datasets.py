import numpy as np
import pandas as pd
import os
from utils_train_test_data import custom_train_test_split, normalize_coverage_per_gene, replace_ones, gaussian_smooth_profiles

##################### Dataset information #####################
# Make sure to only combine datasets with the same window and binsize
datasets = ['Btheta_3200_1600_TU_norm', 'Diet1_3200_1600_TU_norm', 'Diet2_3200_1600_TU_norm','Paraquat_3200_1600_TU_norm']
combined_dataset_name = "3200_1600_TU_norm"
window_size = 3200
overlap_size = 1600
###############################################################

data_dir = '/cluster/scratch/hugifl/spacer_coverage_final_data_2/'
outdir = data_dir  + combined_dataset_name + "_data/"

if not os.path.exists(outdir):
    os.makedirs(outdir)

total_library_size = 0
library_sizes = []

print("Combining datasets: ", datasets)
print("Extracting library sizes...")
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
    print("Processing dataset: ", dataset_name)
    data_file = data_dir + dataset_name + "_data/" + "XY_data_Y_with_windows.npz"
    data = np.load(data_file)
    X = data['X']
    X = X.astype(np.float32)
    Y = data['Y']

    # Count the total number of elements
    total_elements = Y.size

    # Check for NaN values and count them
    nan_count = np.sum(np.isnan(Y))

    # Check for Inf values and count them
    inf_count = np.sum(np.isinf(Y))

    # Calculate the total count of NaN or Inf values
    total_nan_inf = nan_count + inf_count

    # Calculate the ratio of NaN or Inf values
    ratio_nan_inf = total_nan_inf / total_elements
    print(f"Ratio of NaN or Inf values in dataset {dataset_name}: {ratio_nan_inf:.4f}")
    
    
    # Calculate weight for this dataset
    weight = library_sizes[index] / total_library_size
    
    # Weight the coverage data
    weighted_coverage = Y[:, 2:] * weight
    #print("example coverage: ", Y[500,1000:1050])
    #print("max Y: ", np.max(Y[:,2:]))
    #print("max weighted_coverage: ", np.max(weighted_coverage))
    # Add to combined coverage
    if combined_coverage is None:
        combined_coverage = np.zeros_like(weighted_coverage)
    combined_coverage += weighted_coverage

# Combined coverage now contains the summed coverage, weighted by library size
# The window information (first two columns) can be added back if necessary
final_Y = np.hstack((Y[:, :2], combined_coverage))

print("Original shape of X: ", X.shape)
print("Original shape of final_Y: ", final_Y.shape)

# Create a mask for rows with NaN or Inf values in final_Y
invalid_rows_mask = np.any(np.isnan(final_Y) | np.isinf(final_Y), axis=1)

# Count the number of rows to be removed
rows_to_remove = np.sum(invalid_rows_mask)
print(f"Number of rows to remove: {rows_to_remove}")

# Remove the rows from X and final_Y
X = X[~invalid_rows_mask, :]
final_Y = final_Y[~invalid_rows_mask, :]

# Print the new shapes of X and final_Y
print("New shape of X after removing invalid rows: ", X.shape)
print("New shape of final_Y after removing invalid rows: ", final_Y.shape)


print("Saving combined dataset...")
final_Y_smoothed = gaussian_smooth_profiles(final_Y, sigma=3)
np.savez(outdir+'XY_data_Y_with_windows_smoothed.npz', X=X, Y=final_Y_smoothed)


X_train, X_test, Y_train, Y_test = custom_train_test_split(X, final_Y_smoothed, window_size, overlap_size, 0.05,  random_state=None)
np.savez(outdir + 'train_test_data_normalized_windows_info_smoothed.npz', X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test)

np.savez(outdir+'XY_data_Y_with_window.npz', X=X, Y=final_Y)

X_train, X_test, Y_train, Y_test = custom_train_test_split(X, final_Y, window_size, overlap_size, 0.05,  random_state=None)
np.savez(outdir + 'train_test_data_normalized_windows_info.npz', X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test)