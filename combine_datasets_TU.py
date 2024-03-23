import numpy as np
import pandas as pd
import os
from utils_train_test_data import custom_train_test_split_TU, normalize_coverage_per_gene, replace_ones, gaussian_smooth_profiles, scale_to_0_1, scale_to_0_1_global, scale_to_0_1_global_with_max

##################### Dataset information #####################
# Make sure to only combine datasets with the same window and binsize
datasets = ['Btheta_Transcriptional_Units_TU_norm_V2_data', 'Diet1_Transcriptional_Units_TU_norm_V2_data', 'Diet2_Transcriptional_Units_TU_norm_V2_data','Paraquat_Transcriptional_Units_TU_norm_V2_data']
combined_dataset_name = "Transcriptional_Units_TU_norm_V2_data_2"

###############################################################

data_dir = '/cluster/scratch/hugifl/spacer_coverage_final_data_2/'
outdir = data_dir  + combined_dataset_name + "/"
pad_symbol = 0.42
#load spacer count file of first dataset
spacer_count_file = data_dir + datasets[0]  + '/spacer_counts_TU.csv'
spacer_count_df = pd.read_csv(spacer_count_file)

if not os.path.exists(outdir):
    os.makedirs(outdir)

total_library_size = 0
library_sizes = []

print("Combining datasets: ", datasets)
print("Extracting library sizes...")
# Calculate total library size and store individual sizes
for dataset_name in datasets:
    library_size_file = data_dir + dataset_name + "/" + 'tot_number_aligned_reads.txt'
    with open(library_size_file, 'r') as file:
        library_size = int(file.read().strip())
        total_library_size += library_size
        library_sizes.append(library_size)

print("Extracting low expressed TUs ...")
low_TU_df_total = pd.DataFrame()
for dataset_name in datasets:
    low_TU_file = data_dir + dataset_name + "/" + 'low_expressed_TUs.csv'
    low_TU_df = pd.read_csv(low_TU_file)
    low_TU_df_total = pd.concat([low_TU_df_total, low_TU_df], axis=0)
# remove multiplicates in rows
print("shape of low_TU_df_total before removing duplicates: ", low_TU_df_total.shape)
print("head of low_TU_df_total before removing duplicates: ", low_TU_df_total.head())
low_TU_df_total = low_TU_df_total.drop_duplicates(keep='first')
print("number of low expressed TUs in total: ", len(low_TU_df_total))
total_low_TU = len(low_TU_df_total)
print("shape of low_TU_df_total: ", low_TU_df_total.shape)

# Initialize empty array for combined coverage
combined_coverage = None

# Process each dataset
for idx, dataset_name in enumerate(datasets):
    print("Processing dataset: ", dataset_name)
    data_file = data_dir + dataset_name + "/" + "XY_data_Y_with_windows.npz"
    data = np.load(data_file)
    X = data['X']
    X = X.astype(np.float32)
    Y = data['Y']

    # iterate through low expressed TUs
    counter = 0

    for index, row in low_TU_df_total.iterrows():
        #print("fraction of low expressed TUs processed: ", counter/total_low_TU)
        TU_name = row['0']
        # find start and end of TU using spacer_count_df
        TU_start = spacer_count_df.loc[spacer_count_df['TU_Name'] == TU_name, 'start'].iloc[0]
        TU_end = spacer_count_df.loc[spacer_count_df['TU_Name'] == TU_name, 'end'].iloc[0]

        #  find the rows in Y that correspond to the TU
        TU_rows = np.where((Y[:,0] == TU_start) & (Y[:,1] == TU_end))
        if len(TU_rows[0]) > 0:
            #print("removed a TU")
            #remove the rows from Y
            Y = np.delete(Y, TU_rows, axis=0)
            X = np.delete(X, TU_rows, axis=0)
        counter += 1
        
    print("start and end sites of first TUs: ", Y[:10,0], Y[:10,1])
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
    weight = library_sizes[idx] / total_library_size
    
    # Weight the coverage data
    weighted_coverage = Y[:, 2:] * weight
    #print("example coverage: ", Y[500,1000:1050])
    #print("max Y: ", np.max(Y[:,2:]))
    #print("max weighted_coverage: ", np.max(weighted_coverage))
    # Add to combined coverage
    if combined_coverage is None:
        combined_coverage = np.zeros_like(weighted_coverage)
    combined_coverage += weighted_coverage

scaled_coverage = scale_to_0_1(combined_coverage)
scaled_coverage_global = scale_to_0_1_global_with_max(combined_coverage, max_value=8)#scale_to_0_1_global(combined_coverage)
# Combined coverage now contains the summed coverage, weighted by library size
# The window information (first two columns) can be added back if necessary

final_Y = np.hstack((Y[:, :2], combined_coverage))
final_Y_scaled = np.hstack((Y[:, :2], scaled_coverage))
final_Y_scaled_global = np.hstack((Y[:, :2], scaled_coverage_global))


# Create a mask for rows with NaN or Inf values in final_Y
invalid_rows_mask = np.any(np.isnan(final_Y) | np.isinf(final_Y), axis=1)
invalid_rows_mask_scaled = np.any(np.isnan(final_Y_scaled) | np.isinf(final_Y_scaled), axis=1)
invalid_rows_mask_scaled_global = np.any(np.isnan(final_Y_scaled_global) | np.isinf(final_Y_scaled_global), axis=1)


# Count the number of rows to be removed
rows_to_remove = np.sum(invalid_rows_mask)
print(f"Number of rows to remove: {rows_to_remove}")
rows_to_remove_scaled = np.sum(invalid_rows_mask_scaled)
print(f"Number of rows to remove in scaled data: {rows_to_remove_scaled}")
rows_to_remove_scaled_global = np.sum(invalid_rows_mask_scaled_global)
print(f"Number of rows to remove in scaled data: {rows_to_remove_scaled_global}")
# Remove the rows from X and final_Y
X = X[~invalid_rows_mask, :]
final_Y = final_Y[~invalid_rows_mask, :]

X_scaled = X[~invalid_rows_mask_scaled, :]
final_Y_scaled = final_Y_scaled[~invalid_rows_mask_scaled, :]

X_scaled_global = X[~invalid_rows_mask_scaled_global, :]
final_Y_scaled_global = final_Y_scaled_global[~invalid_rows_mask_scaled_global, :]
# Print the new shapes of X and final_Y
print("New shape of X after removing invalid rows: ", X.shape)
print("New shape of final_Y after removing invalid rows: ", final_Y.shape)
print("New shape of X_scaled after removing invalid rows: ", X_scaled.shape)
print("New shape of final_Y_scaled after removing invalid rows: ", final_Y_scaled.shape)
print("New shape of X_scaled after removing invalid rows: ", X_scaled_global.shape)
print("New shape of final_Y_scaled after removing invalid rows: ", final_Y_scaled_global.shape)
print("Saving combined dataset...")
final_Y_smoothed = gaussian_smooth_profiles(final_Y, sigma=3)
np.savez(outdir+'XY_data_Y_with_windows_smoothed.npz', X=X, Y=final_Y_smoothed)


X_train, X_test, Y_train, Y_test = custom_train_test_split_TU(X, final_Y_smoothed, 0.1,  random_state=None)
np.savez(outdir + 'train_test_data_normalized_windows_info_smoothed.npz', X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test)

np.savez(outdir+'XY_data_Y_with_window.npz', X=X, Y=final_Y)

X_train, X_test, Y_train, Y_test = custom_train_test_split_TU(X, final_Y, 0.1,  random_state=None)
np.savez(outdir + 'train_test_data_normalized_windows_info.npz', X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test)

print("Saving locally scaled combined dataset...")
final_Y_scaled_smoothed = gaussian_smooth_profiles(final_Y_scaled, sigma=3)
np.savez(outdir+'XY_data_Y_with_windows_scaled_smoothed.npz', X=X_scaled, Y=final_Y_scaled_smoothed)

X_train, X_test, Y_train, Y_test = custom_train_test_split_TU(X_scaled, final_Y_scaled_smoothed, 0.1,  random_state=None)
np.savez(outdir + 'train_test_data_normalized_windows_info_scaled_smoothed.npz', X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test)

X_train, X_test, Y_train, Y_test = custom_train_test_split_TU(X_scaled, final_Y_scaled, 0.1,  random_state=None)
np.savez(outdir + 'train_test_data_normalized_windows_info_scaled.npz', X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test)

print("Saving globally scaled combined dataset...")
final_Y_scaled_smoothed = gaussian_smooth_profiles(final_Y_scaled_global, sigma=3)
np.savez(outdir+'XY_data_Y_with_windows_scaled_global_smoothed.npz', X=X_scaled_global, Y=final_Y_scaled_smoothed)

X_train, X_test, Y_train, Y_test = custom_train_test_split_TU(X_scaled_global, final_Y_scaled_smoothed, 0.1,  random_state=None)
np.savez(outdir + 'train_test_data_normalized_windows_info_scaled_global_smoothed.npz', X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test)

X_train, X_test, Y_train, Y_test = custom_train_test_split_TU(X_scaled_global, final_Y_scaled_global, 0.1,  random_state=None)
np.savez(outdir + 'train_test_data_normalized_windows_info_scaled_global.npz', X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test)