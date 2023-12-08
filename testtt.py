
import pandas as pd
from utils_sequence import parse_fasta, one_hot_encode, extract_sequences_and_sequence_info, dataframe_to_2darray_keep_window_information
from utils_coverage import gaussian_smooth_profiles
from utils_plotting import plot_window_coverage_normalized
import numpy as np

##################### Set before training #####################

window_size = 3200
overlap = 1600
no_bin = 1600
binsize = 2
no_plots = 40
dataset_name = 'window_3200_overlapt_1600_binsize_2'

###############################################################

tsv_file = "/cluster/home/hugifl/spacer_coverage_input/ECOCYC_genes.txt"
tsv_file2 = "/cluster/home/hugifl/spacer_coverage_input/ECOCYC_promoters.txt"
tsv_file3 = "/cluster/home/hugifl/spacer_coverage_input/ECOCYC_terminators.txt"
coverage_df_file = "/cluster/home/hugifl/spacer_coverage_output/test_outputs/window_coverage_data_summed.csv"
genome_file = "/cluster/home/hugifl/exon_coverage_input_output/U00096.3.fasta"
outfile = "/cluster/home/hugifl/spacer_coverage_output/"
train_test_data_file = "/cluster/scratch/hugifl/spacer_coverage_final_data/window_3200_overlapt_1600_binsize_2_2_data/train_test_data_normalized_windows_info_.npz"
outdir = "/cluster/scratch/hugifl/spacer_coverage_final_data/window_3200_overlapt_1600_binsize_2_2_data/"
promoter_df = pd.read_csv(tsv_file2, sep='\t')
promoter_df.dropna(inplace=True)

terminator_df = pd.read_csv(tsv_file3, sep='\t')
terminator_df.dropna(inplace=True)


gene_df = pd.read_csv(tsv_file, sep='\t')
gene_df.drop(gene_df.columns[1], axis=1, inplace=True)
gene_df.dropna(inplace=True)
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

#scaling_factor = 0.5
#
#Y_test[:,2:] = Y_test[:,2:] * scaling_factor
#Y_train[:,2:] = Y_train[:,2:] * scaling_factor
#Y_tot = np.vstack((Y_test, Y_train))
#Y_tot = Y_tot
#
#Y_tot_smoothed = gaussian_smooth_profiles(Y_tot, sigma=3)



Y_test_smoothed = gaussian_smooth_profiles(Y_test, sigma=3)
Y_train_smoothed = gaussian_smooth_profiles(Y_train, sigma=3)

np.savez(outdir + 'train_test_data_normalized_windows_info_smoothed.npz', X_train=X_train, X_test=X_test, Y_train=Y_train_smoothed, Y_test=Y_test_smoothed)

#plots_normalized = plot_window_coverage_normalized(Y_tot_smoothed, no_plots, no_bin, outdir, dataset_name, window_size,promoter_df, terminator_df, gene_df, binsize, random=False)
#print(plots_normalized)
