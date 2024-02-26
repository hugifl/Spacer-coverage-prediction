import numpy as np

gene_dataset = '3200_1600_gene_norm'
TU_dataset = '3200_1600_TU_norm'

outdir = '../spacer_coverage_output_2/'
datadir = '/cluster/scratch/hugifl/spacer_coverage_final_data_2/'
gene_data_file = datadir + gene_dataset + "_data"+"/train_test_data_normalized_windows_info.npz"
TU_data_file = datadir + TU_dataset + "_data"+"/train_test_data_normalized_windows_info.npz"

gene_data = np.load(gene_data_file)
X_train_gene = gene_data['X_train']
X_test_gene = gene_data['X_test']
Y_train_gene = gene_data['Y_train']
Y_test_gene = gene_data['Y_test']

TU_data = np.load(TU_data_file)
X_train_TU = TU_data['X_train']
X_test_TU = TU_data['X_test']
Y_train_TU = TU_data['Y_train']
Y_test_TU = TU_data['Y_test']

# Find rows with NaNs or Infs in Y_train
rows_with_nans_or_infs_train_gene = np.any(np.isnan(Y_train_gene) | np.isinf(Y_train_gene), axis=1)
Y_train_filtered_gene = Y_train_gene[~rows_with_nans_or_infs_train_gene]
X_train_filtered_gene = X_train_gene[~rows_with_nans_or_infs_train_gene]

# Find rows with NaNs or Infs in Y_test
rows_with_nans_or_infs_test = np.any(np.isnan(Y_test_gene) | np.isinf(Y_test_gene), axis=1)
Y_test_filtered = Y_test_gene[~rows_with_nans_or_infs_test]
X_test_filtered = X_test_gene[~rows_with_nans_or_infs_test]

print("gene_norm: number of rows with nans or infs in Y_train: ", np.sum(rows_with_nans_or_infs_train_gene))
print("gene_norm: number of rows with nans or infs in Y_test: ", np.sum(rows_with_nans_or_infs_test))

rows_with_nans_or_infs_train_TU = np.any(np.isnan(Y_train_TU) | np.isinf(Y_train_TU), axis=1)
Y_train_filtered_TU = Y_train_TU[~rows_with_nans_or_infs_train_TU]
X_train_filtered_TU = X_train_TU[~rows_with_nans_or_infs_train_TU]

rows_with_nans_or_infs_test_TU = np.any(np.isnan(Y_test_TU) | np.isinf(Y_test_TU), axis=1)
Y_test_filtered_TU = Y_test_TU[~rows_with_nans_or_infs_test_TU]
X_test_filtered_TU = X_test_TU[~rows_with_nans_or_infs_test_TU]

print("TU_norm: number of rows with nans or infs in Y_train: ", np.sum(rows_with_nans_or_infs_train_TU))
print("TU_norm: number of rows with nans or infs in Y_test: ", np.sum(rows_with_nans_or_infs_test_TU))

print("average max coverage in gene_norm test: ", np.mean(np.max(Y_test_filtered[:,2:], axis=1)))
print("standard deviation of max coverage in gene_norm test: ", np.std(np.max(Y_test_filtered[:,2:], axis=1)))
print("average max coverage in TU_norm test: ", np.mean(np.max(Y_test_filtered_TU[:,2:], axis=1)))
print("standard deviation of max coverage in TU_norm test: ", np.std(np.max(Y_test_filtered_TU[:,2:], axis=1)))
print("average max coverage in gene_norm train: ", np.mean(np.max(Y_train_filtered_gene[:,2:], axis=1)))
print("standard deviation of max coverage in gene_norm train: ", np.std(np.max(Y_train_filtered_gene[:,2:], axis=1)))
print("average max coverage in TU_norm train: ", np.mean(np.max(Y_train_filtered_TU[:,2:], axis=1)))
print("standard deviation of max coverage in TU_norm train: ", np.std(np.max(Y_train_filtered_TU[:,2:], axis=1)))
print("-------------------------------------")
print("average coverage in gene_norm test: ", np.mean(np.mean(Y_test_filtered[:,2:], axis=1)))
print("standard deviation of coverage in gene_norm test: ", np.std(np.mean(Y_test_filtered[:,2:], axis=1)))
print("average coverage in TU_norm test: ", np.mean(np.mean(Y_test_filtered_TU[:,2:], axis=1)))
print("standard deviation of coverage in TU_norm test: ", np.std(np.mean(Y_test_filtered_TU[:,2:], axis=1)))
print("average coverage in gene_norm train: ", np.mean(np.mean(Y_train_filtered_gene[:,2:], axis=1)))
print("standard deviation of coverage in gene_norm train: ", np.std(np.mean(Y_train_filtered_gene[:,2:], axis=1)))
print("average coverage in TU_norm train: ", np.mean(np.mean(Y_train_filtered_TU[:,2:], axis=1)))
print("standard deviation of coverage in TU_norm train: ", np.std(np.mean(Y_train_filtered_TU[:,2:], axis=1)))