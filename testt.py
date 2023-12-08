
import csv
from utils_sequence import parse_fasta, one_hot_encode, extract_sequences_and_sequence_info, dataframe_to_2darray_keep_window_information
from utils_coverage import gaussian_smooth_profiles
import numpy as np
tsv_file = "/cluster/home/hugifl/spacer_coverage_input/ECOCYC_genes.txt"
tsv_file2 = "/cluster/home/hugifl/spacer_coverage_input/ECOCYC_promoters.txt"
tsv_file3 = "/cluster/home/hugifl/spacer_coverage_input/ECOCYC_terminators.txt"
coverage_df_file = "/cluster/home/hugifl/spacer_coverage_output/test_outputs/window_coverage_data_summed.csv"
genome_file = "/cluster/home/hugifl/exon_coverage_input_output/U00096.3.fasta"
outfile = "/cluster/home/hugifl/recordseq-workflow-dev/dev-hugi/exon_coverage/output/gene_coverage_scaled.csv"
train_test_data_file = "/cluster/home/hugifl/spacer_coverage_output/window_3200_overlapt_1600_binsize_2_outputs/train_test_data_normalized_windows_info_.npz"
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

scaling_factor = 1e-7

Y_test[:,2:] = Y_test[:,2:] * scaling_factor
Y_train[:,2:] = Y_train[:,2:] * scaling_factor
Y_tot = np.vstack((Y_test, Y_train))
Y_tot = Y_tot[:,2:]

import matplotlib.pyplot as plt
import seaborn as sns

data =  pd.read_csv(coverage_df_file, sep=',', comment="#")
data = data.to_numpy()
# Assuming 'data' is your dataset as a numpy array of shape (no_windows, bins_per_window)
# Flatten the array to get a 1D array of all coverage values
coverage_values = Y_tot.flatten()

plt.figure(figsize=(10, 6))
plt.hist(coverage_values, bins=750, density=True)
plt.xlabel('Read Coverage')
plt.xlim(xmin=0, xmax=5) 
plt.ylabel('Density')
plt.title('Read Coverage Distribution')
plt.savefig('/cluster/home/hugifl/spacer_coverage_output/window_3200_overlapt_1600_binsize_2_outputs/'+ "spacer_coverage_distribution.png")
plt.close()

#plt.figure(figsize=(10, 6))
#sns.kdeplot(coverage_values, bw_adjust=0.5)  # Ensure sns.kdeplot is used
#plt.xlabel('Read Coverage')
#plt.ylabel('Density')
#plt.title('Read Coverage Distribution')
#plt.savefig('/cluster/home/hugifl/spacer_coverage_output/window_3200_overlapt_1600_binsize_2_outputs/'+ "spacer_coverage_distribution.png")
#plt.close()
