
import numpy as np


data = np.load('../exon_coverage_input_output/output/train_test_data_normalized_windows_info_'+str(3200) + '_' + str(1602) + '.npz')
X_train = data['X_train']
X_test = data['X_test']
Y_train = data['Y_train']
Y_test = data['Y_test']

# Count rows with NaNs
#rows_with_nans = np.sum(np.any(np.isnan(Y_test), axis=1))
#
## Rows with NaNs
#nan_rows = np.where(np.any(np.isnan(Y_train), axis=1))[0]
#
## Rows with Infs
#inf_rows = np.where(np.any(np.isinf(Y_train), axis=1))[0]
#
#print(f"Rows with NaNs: {nan_rows}")
#print(f"Rows with Infs: {inf_rows}")

print(Y_train[4,:50])