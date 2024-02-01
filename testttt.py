import pandas as pd
import numpy as np  
from utils_coverage import filter_bamlist, total_count_per_bam

para_total_counts = pd.read_csv('/cluster/scratch/hugifl/paraquat_run_1/outputs/countsMatrices/totalCounts.txt', sep='\t')
diet_total_counts = pd.read_csv('/cluster/scratch/hugifl/paraquat_run_2/outputs/countsMatrices/totalCounts.txt', sep='\t')

print("number of samples in para: ", para_total_counts.shape[0])
print("number of samples in diet: ", diet_total_counts.shape[0])
filtered_para_total_counts = para_total_counts[para_total_counts.iloc[:, 2] >= 10000]
filtered_diet_total_counts = diet_total_counts[diet_total_counts.iloc[:, 2] >= 10000]
print("number of samples in para with at least 10000 genome counts: ", filtered_para_total_counts.shape[0])
print("number of samples in diet with at least 10000 genome counts: ", filtered_diet_total_counts.shape[0])

print("average reads per sample in para: ", para_total_counts.iloc[:, 1].mean())
print("average reads per sample in diet: ", diet_total_counts.iloc[:, 1].mean())

print("average genome counts per sample in para: ", para_total_counts.iloc[:, 2].mean())
print("average genome counts per sample in diet: ", diet_total_counts.iloc[:, 2].mean())

print("average genome counts per reads in a sample in para: ", para_total_counts.iloc[:, 2].mean()/para_total_counts.iloc[:, 1].mean())
print("average genome counts per reads in a sample in diet: ", diet_total_counts.iloc[:, 2].mean()/diet_total_counts.iloc[:, 1].mean())

para_coverage_df = pd.read_csv('/cluster/scratch/hugifl/spacer_coverage_final_data/Paraquat_window_3200_overlap_1600_no_binning_gene_normalized_data/window_coverage_data_summed.csv')
diet_coverage_df = pd.read_csv('/cluster/scratch/hugifl/spacer_coverage_final_data/Diet1_window_3200_overlap_1600_no_binning_gene_normalized_data/window_coverage_data_summed.csv')
print("mean para_coverage_df: ", para_coverage_df.iloc[:, 2:].mean().mean())
print("mean diet_coverage_df: ", diet_coverage_df.iloc[:, 2:].mean().mean())


count_df_para = pd.read_csv('/cluster/home/hugifl/spacer_coverage_input/genomeCounts_paraquat.txt', sep='\t')
count_df_diet = pd.read_csv('/cluster/home/hugifl/spacer_coverage_input/genomeCounts_diet1.txt', sep='\t')

print('mean spacer counts per gene in para: ', count_df_para.iloc[:, 6:].mean().mean())
print('mean spacer counts per gene in diet: ', count_df_diet.iloc[:, 6:].mean().mean())

para_coverage = '/cluster/scratch/hugifl/spacer_coverage_final_data/Paraquat_window_3200_overlap_1600_no_binning_gene_normalized_data/XY_data_Y_with_windows.npz'
para_coverage= np.load(para_coverage)

diet_coverage = '/cluster/scratch/hugifl/spacer_coverage_final_data/Diet1_window_3200_overlap_1600_no_binning_gene_normalized_data/XY_data_Y_with_windows.npz'
diet_coverage= np.load(diet_coverage)

para_coverage = para_coverage['Y']
diet_coverage = diet_coverage['Y']

para_coverage = para_coverage[:,2:]
diet_coverage = diet_coverage[:,2:]

print(" mean para_coverage: ", np.mean(para_coverage))
print(" mean diet_coverage: ", np.mean(diet_coverage))

para_avg_gene_length = count_df_para['Length'].mean()
diet_avg_gene_length = count_df_diet['Length'].mean()
print("para_avg_gene_length: ", para_avg_gene_length)
print("diet_avg_gene_length: ", diet_avg_gene_length)

def load_file_and_compute_average(file_path):
    # Load the file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Check if 'spacer_counts' column exists
    if 'spacer_count' in df.columns:
        # Calculate the average of the column 'spacer_counts'
        average = df['spacer_count'].mean()
        print(f"The average of 'spacer_count' is: {average}")
    else:
        print("The column 'spacer_counts' does not exist in the data.")
    return average

# Replace 'path/to/your/file.txt' with the actual path to your text file
file_path_para = '/cluster/scratch/hugifl/spacer_coverage_final_data/Paraquat_window_3200_overlap_1600_no_binning_gene_normalized_data/gene_spacer_counts.csv'
library_para = 12435391
avg_para = load_file_and_compute_average(file_path_para)
print("avg_para per library: ", avg_para/library_para)


file_path_diet = '/cluster/scratch/hugifl/spacer_coverage_final_data/Diet1_window_3200_overlap_1600_no_binning_gene_normalized_data/gene_spacer_counts.csv'
library_diet = 23447135
avg_diet = load_file_and_compute_average(file_path_diet)
print("avg_diet per library: ", avg_diet/library_diet)

para_diet_ratio = avg_para/avg_diet

print("para_diet_ratio: ", para_diet_ratio)

print("------------------ para ------------------")
count_dict_para = total_count_per_bam(count_df_para, 'S') 
print("number of bams in para: ", len(count_dict_para))
bamlist_para = list(count_dict_para.keys())
bamlist_para = ['S' + s.split('S', 1)[1] if 'S' in s else s for s in bamlist_para] 
print("unfiltered number of bam files: ", len(bamlist_para))
bamlist = filter_bamlist(bamlist_para, count_df_para, 10000, 'S') # Unnecessary if we just add them up anyways
print("filtered number of bam files: ",len(bamlist))

print("------------------ diet ------------------")
count_dict_diet = total_count_per_bam(count_df_diet, 'BSSE')
print("number of bams in diet: ", len(count_dict_diet))
bamlist_diet = list(count_dict_diet.keys())
bamlist_diet = ['BSSE' + s.split('BSSE', 1)[1] if 'BSSE' in s else s for s in bamlist_diet]
print("unfiltered number of bam files: ", len(bamlist_diet))
bamlist = filter_bamlist(bamlist_diet, count_df_diet, 10000, 'BSSE') # Unnecessary if we just add them up anyways
print("filtered number of bam files: ",len(bamlist))