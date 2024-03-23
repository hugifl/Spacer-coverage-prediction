import numpy as np
import pandas as pd
import os
from scipy.stats import pearsonr

##################### Dataset information #####################
# Make sure to only combine datasets with the same window and binsize
datapath = '/cluster/scratch/hugifl/spacer_coverage_final_data_2/data_for_correlations/'

B_theta_file = datapath + 'B_theta_coverage.csv'
Diet1_file = datapath + 'Diet1_coverage.csv'
Diet2_file = datapath + 'Diet2_coverage.csv'
Paraquat_file = datapath + 'Paraquat_coverage.csv'

B_theta_df = pd.read_csv(B_theta_file)
Diet1_df = pd.read_csv(Diet1_file)
Diet2_df = pd.read_csv(Diet2_file)
Paraquat_df = pd.read_csv(Paraquat_file)



dfs = [B_theta_df, Diet1_df, Diet2_df, Paraquat_df]
df_names = ['B_theta', 'Diet1', 'Diet2', 'Paraquat']
print("shape of B_theta_df: ", B_theta_df.shape)
print("shape of Diet1_df: ", Diet1_df.shape)
print("shape of Diet2_df: ", Diet2_df.shape)
print("shape of Paraquat_df: ", Paraquat_df.shape)
rows_to_remove = set()
for df in dfs:
    # Find indices of rows with NaN or Inf in this df
    nan_inf_indices = df.index[df.isin([np.nan, np.inf, -np.inf]).any(axis=1)]
    rows_to_remove.update(nan_inf_indices)

# Convert to list for iloc
rows_to_remove = sorted(list(rows_to_remove))

# Step 2 & 3: Remove identified rows from each df
dfs_clean = [df.drop(rows_to_remove, errors='ignore') for df in dfs]
B_theta_df, Diet1_df, Diet2_df, Paraquat_df = dfs_clean
print("shape of B_theta_df: ", B_theta_df.shape)
print("shape of Diet1_df: ", Diet1_df.shape)
print("shape of Diet2_df: ", Diet2_df.shape)
print("shape of Paraquat_df: ", Paraquat_df.shape)
# Function to calculate pairwise Pearson correlations
def calculate_correlations(dfs, df_names):
    correlations = []
    for i, df1 in enumerate(dfs):
        for j, df2 in enumerate(dfs):
            if j <= i:  
                continue
            print("new pair")
            corrs = [pearsonr(df1.iloc[row, 2:], df2.iloc[row, 2:])[0] for row in range(df1.shape[0])]
            corrs = [c for c in corrs if not np.isnan(c)]  
            correlations.append(corrs)
            print(f"Pair: {df_names[i]} - {df_names[j]} | Mean: {np.mean(corrs):.3f}, Max: {np.max(corrs):.3f}, Min: {np.min(corrs):.3f}")
    # Calculate the average of all correlations
    all_corrs = [c for sublist in correlations for c in sublist]  # Flatten the list
    print(f"Average correlation across all pairs: {np.mean(all_corrs):.3f}")
    return all_corrs

all_corrs = calculate_correlations(dfs_clean, df_names)
print(f"Average correlation across all pairs: {np.mean(all_corrs):.3f}")