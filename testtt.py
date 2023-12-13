
import pandas as pd
from utils_sequence import parse_fasta, one_hot_encode, extract_sequences_and_sequence_info, dataframe_to_2darray_keep_window_information

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
count_file = "/cluster/home/hugifl/spacer_coverage_input/genomeCounts_UMI_paraquat.txt"
promoter_df = pd.read_csv(tsv_file2, sep='\t')
promoter_df.dropna(inplace=True)

terminator_df = pd.read_csv(tsv_file3, sep='\t')
terminator_df.dropna(inplace=True)

count_df = pd.read_csv(count_file, sep='\t')
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

# Define the filename
data = np.load(train_test_data_file)
X_train = data['X_train']
X_test = data['X_test']
Y_train = data['Y_train']
Y_test = data['Y_test']
# Adjust the coverage data

import pandas as pd

# Example DataFrame creation
# df = pd.read_csv('your_file.csv')  # Uncomment this line to read from a CSV file



# Function to process column names
def process_column_name(col_name):
    if col_name.startswith('/cluster/scratch/hugifl/paraquat_run_1/outputs_umi/alignments/genomeBams/'):
        # Split on 'S' and keep the part including and after 'S'
        return 'S' + col_name.split('S', 1)[1]
    return col_name

# Apply the function to each column name
count_df.columns = [process_column_name(col) for col in count_df.columns]
# Assuming your DataFrame is named df
count_df.to_csv('/cluster/home/hugifl/spacer_coverage_input/genomeCounts_UMI_paraquat2.txt', sep='\t', index=False)

