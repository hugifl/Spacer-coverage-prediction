import numpy as np
import numpy as np
from utils_train_test_data import scale_to_0_1_global_with_max

np.random.seed(42)  # For reproducibility
num_profiles = 5  # Number of coverage profiles
profile_length = 300  # Length of each profile
padding_length = 100  # Length of padding

data_file = '/cluster/scratch/hugifl/spacer_coverage_final_data_2/Diet2_Transcriptional_Units_TU_norm_V2_data/XY_data_Y_with_windows.npz'
data = np.load(data_file)
X = data['X']
X = X.astype(np.float32)
Y = data['Y']

# Create random data for the non-padding part and pad the rest

print("head of Y: ", Y[:, 2:] [:5, :5])
print("max of Y: ", np.max(Y))
print("index of max of Y: ", np.argmax(Y))
print("end of Y: ", Y[:, 2:] [-5:, -5:])
# Scale the profiles globally
Y_scaled = scale_to_0_1_global_with_max(Y[:, 2:] , max_value=8)

# Print some of the scaled data to inspect

print("head of Y_scaled: ", Y_scaled[:5, :5])
print("max of Y_scaled: ", np.max(Y_scaled))
print("index of max of Y_scaled: ", np.argmax(Y_scaled))
print("end of Y: ", Y_scaled[-5:, -5:])

# plot the first row of the scaled data
import matplotlib.pyplot as plt
plt.plot(Y_scaled[0, :500])
plt.savefig('/cluster/home/hugifl/spacer_coverage_prediction/testplots/spacer_coverage_scaled.png')
plt.close()
plt.plot(Y[0, 2:500])
plt.savefig('/cluster/home/hugifl/spacer_coverage_prediction/testplots/spacer_coverage.png')
plt.close()