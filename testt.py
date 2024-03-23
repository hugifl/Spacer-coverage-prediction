
import csv
from utils_sequence import parse_fasta, one_hot_encode, extract_sequences_and_sequence_info, dataframe_to_2darray_keep_window_information
from scipy.stats import chisquare, poisson
import numpy as np
import matplotlib.pyplot as plt

train_test_data_file = "/cluster/scratch/hugifl/spacer_coverage_final_data_2/3200_1600_gene_norm_data/train_test_data_normalized_windows_info.npz"
# Open the GTF file for reading
#counter = 0
#maximum = 0
#for a in gtf_file:
#    end = max(a.iv.start_d_as_pos.pos, a.iv.end_d_as_pos.pos)
#    if end > maximum:
#        maximum = end
#
#print(maximum)
import pandas as pd

# Define the filename
data = np.load(train_test_data_file)
X_train = data['X_train']
X_test = data['X_test']
Y_train = data['Y_train']
Y_test = data['Y_test']
# Adjust the coverage data



Y_test[:,2:] = Y_test[:,2:] 
Y_train[:,2:] = Y_train[:,2:] 
Y_tot = np.vstack((Y_test, Y_train))
Y_tot = Y_tot[:,2:]

# Assuming 'data' is your dataset as a numpy array of shape (no_windows, bins_per_window)
# Flatten the array to get a 1D array of all coverage values
coverage_values = Y_tot.flatten()

mean_coverage = np.mean(coverage_values)
variance_coverage = np.var(coverage_values)

print(f"Mean of coverage values: {mean_coverage}")
print(f"Variance of coverage values: {variance_coverage}")



plt.figure(figsize=(10, 6))
plt.hist(coverage_values, bins=750, density=True)
plt.xlabel('Read Coverage')
plt.xlim(xmin=0, xmax=5) 
plt.ylabel('Density')
plt.title('Read Coverage Distribution')
plt.savefig('/cluster/home/hugifl/spacer_coverage_output_2/3200_1600_gene_norm_outputs/'+ "spacer_coverage_distribution.png")
plt.close()





# Generate Poisson distribution data based on the observed mean
poisson_dist = poisson(mu=mean_coverage)
x_poisson = np.arange(0, np.max(coverage_values), 1)  # Generating a range of values from 0 to max observed value
y_poisson = poisson_dist.pmf(x_poisson) * len(coverage_values)  # Scaling the PMF by the number of observations

# Plot observed data histogram
plt.figure(figsize=(10, 6))
counts, bins, _ = plt.hist(coverage_values, bins=10000, density=False, alpha=0.6, color='g', label='Observed Data')

# Plot Poisson distribution line
plt.plot(x_poisson, y_poisson, 'r-', lw=2, label='Expected Poisson Distribution')
plt.xlim(xmin=0, xmax=5) 
plt.xlabel('Count')
plt.ylabel('Frequency')
plt.title('Count Distribution vs. Poisson Distribution')
plt.legend()
plt.savefig('/cluster/home/hugifl/spacer_coverage_output_2/3200_1600_gene_norm_outputs/'+ "spacer_coverage_distribution2.png")
plt.close()