import pandas
import numpy 
import pandas as pd
import HTSeq
from matplotlib import pyplot
from scipy.ndimage import gaussian_filter1d
from utils_train_test_data import replace_ones
from utils_coverage import total_count_per_bam, filter_bamlist, filter_bamlist_2, get_windows, process_batch, normalize_coverage_per_gene, get_normalized_spacer_counts_per_gene, normalize_coverage_for_tot_aligned_reads
outdir = '/cluster/home/hugifl/spacer_coverage_output_test/'
para_start = 'S'
diet_start = 'BSSE'

# count matrices
paraquat_counts = pd.read_csv('/cluster/home/hugifl/spacer_coverage_input/genomeCounts_paraquat.txt', sep='\t', dtype=str, low_memory=False)
diet_2_counts = pd.read_csv('/cluster/home/hugifl/spacer_coverage_input/genomeCounts_diet2.txt', sep='\t', dtype=str, low_memory=False)

# Convert the datatype of the 7th column onwards to float
paraquat_counts.iloc[:, 6:] = paraquat_counts.iloc[:, 6:].astype(float)
diet_2_counts.iloc[:, 6:] = diet_2_counts.iloc[:, 6:].astype(float)

count_dict_para = total_count_per_bam(paraquat_counts, para_start) # dictionary storing the total spacer counts per bam file
count_dict_diet = total_count_per_bam(diet_2_counts, diet_start) # dictionary storing the total spacer counts per bam file

print("number of bam files in paraquat in count df: ", paraquat_counts.shape[1]-6)
print("number of bam files in diet 2 in count df: ", diet_2_counts.shape[1]-6)
print("number of bam files in paraquat in count dict: ", len(count_dict_para))
print("number of bam files in diet 2 in count dict: ", len(count_dict_diet))

# Assuming count_dict_para is your dictionary
values = list(count_dict_para.values())

# Convert all values to floats
float_values = [float(val) for val in values]

# Calculate the mean
average = numpy.mean(float_values)

print("average number of spacers per bam file in paraquat: ", average)

# Assuming count_dict_para is your dictionary
values = list(count_dict_diet.values())

# Convert all values to floats
float_values = [float(val) for val in values]

# Calculate the mean
average = numpy.mean(float_values)

print("average number of spacers per bam file in diet: ", average)
bamlist_para = list(count_dict_para.keys())
bamlist_diet = list(count_dict_diet.keys())

bamlist_para = filter_bamlist(bamlist_para, paraquat_counts, 10000, para_start)
bamlist_diet = filter_bamlist(bamlist_diet, diet_2_counts, 10000, diet_start)

bamlist_para_2 = filter_bamlist_2(count_dict_para, 10000)
bamlist_diet_2 = filter_bamlist_2(count_dict_diet, 10000)

print("length of filtered bamlist para: ", len(bamlist_para))
print("length of filtered bamlist diet: ", len(bamlist_diet))
print("length of filtered bamlist para 2: ", len(bamlist_para_2))
print("length of filtered bamlist diet 2: ", len(bamlist_diet_2))

count_dict_para = {k: count_dict_para[k] for k in bamlist_para if k in count_dict_para}
count_dict_diet = {k: count_dict_diet[k] for k in bamlist_diet if k in count_dict_diet}


windows = get_windows(4641652,3200,1600)

# Paraquat
print("paraquat")
batch = 0
total_aligned_reads = 0
total_window_aligned_reads = 0
for i in range(0, len(bamlist_para), 5):
    batch += 1
    print("batch " + str(batch) + " out of "+ str(len(bamlist_para)/5))
    bam_batch = bamlist_para[i:i + 5]
    batch_coverage_df, aligned_read_count_batch, window_aligned_read_count_batch = process_batch(bam_batch, windows, 'NC_000913.3', 1, '/cluster/scratch/hugifl/paraquat_run_1/outputs/alignments/genomeBams/') # U00096.3
    total_aligned_reads += aligned_read_count_batch
    total_window_aligned_reads += window_aligned_read_count_batch
    batch_coverage_summed = batch_coverage_df.groupby(['Window_Start', 'Window_End']).sum().reset_index()
    
    if batch == 1:
        coverage_df_summed = batch_coverage_summed
    else:
        # Sum the batch coverage with the total coverage
        coverage_df_summed = pd.concat([coverage_df_summed, batch_coverage_summed]).groupby(['Window_Start', 'Window_End']).sum().reset_index()

total_reads = 0
for bam in bamlist_para:
    bam_file = HTSeq.BAM_Reader(bam)
    for alnmt in bam_file:
        total_reads += 1
print("total aligned reads: ", total_aligned_reads)
print("total window aligned reads: ", total_window_aligned_reads)
print("ratio: ", total_window_aligned_reads/total_aligned_reads)
print("total reads: ", total_reads)
print("ratio aligned to total: ", total_aligned_reads/total_reads)

# Normalize coverage for gene expression (RPKM) values per gene to remove effects of differential gene expression.
low_expressed_genes, gene_spacer_counts_normalized_df = get_normalized_spacer_counts_per_gene(paraquat_counts, 10, count_dict_para)       
coverage_array_summed = coverage_df_summed.to_numpy()
coverage_array_gene_normalized = normalize_coverage_per_gene(coverage_array_summed, gene_spacer_counts_normalized_df, 3200, 1)


# Normalize coverage for total aligned reads to make scale comparable across experiments
coverage_array_gene_and_library_size_normalized = normalize_coverage_for_tot_aligned_reads(coverage_array_gene_normalized, total_aligned_reads, 200000)
coverage_df_gene_and_library_size_normalized = pd.DataFrame(coverage_array_gene_and_library_size_normalized)
coverage_df_gene_and_library_size_normalized.columns = coverage_df_summed.columns

# Save data frame
gene_spacer_counts_normalized_df.to_csv(outdir+'gene_spacer_counts_paraquat.csv', index=False)
coverage_df_gene_and_library_size_normalized.to_csv(outdir+'window_coverage_data_summed_paraquat.csv', index=False)
with open(outdir+'tot_number_aligned_reads.txt', 'w') as file:
    # Write the integer to file
    file.write(str(total_aligned_reads))


# Diet 2
print("diet 2")
batch = 0
total_aligned_reads = 0
total_window_aligned_reads = 0
for i in range(0, len(bamlist_diet), 5):
    batch += 1
    print("batch " + str(batch) + " out of "+ str(len(bamlist_diet)/5))
    bam_batch = bamlist_diet[i:i + 5]
    batch_coverage_df, aligned_read_count_batch, window_aligned_read_count_batch = process_batch(bam_batch, windows, 'U00096.3', 1, '/cluster/scratch/hugifl/gut_diet2_run_1/outputs/alignments/genomeBams/') # U00096.3
    total_aligned_reads += aligned_read_count_batch
    total_window_aligned_reads += window_aligned_read_count_batch
    batch_coverage_summed = batch_coverage_df.groupby(['Window_Start', 'Window_End']).sum().reset_index()
    
    if batch == 1:
        coverage_df_summed = batch_coverage_summed
    else:
        # Sum the batch coverage with the total coverage
        coverage_df_summed = pd.concat([coverage_df_summed, batch_coverage_summed]).groupby(['Window_Start', 'Window_End']).sum().reset_index()


total_reads = 0
for bam in bamlist_diet:
    bam_file = HTSeq.BAM_Reader(bam)
    for alnmt in bam_file:
        total_reads += 1
print("total aligned reads: ", total_aligned_reads)
print("total window aligned reads: ", total_window_aligned_reads)
print("ratio: ", total_window_aligned_reads/total_aligned_reads)
print("total reads: ", total_reads)
print("ratio aligned to total: ", total_aligned_reads/total_reads)



# Normalize coverage for gene expression (RPKM) values per gene to remove effects of differential gene expression.
low_expressed_genes, gene_spacer_counts_normalized_df = get_normalized_spacer_counts_per_gene(diet_2_counts, 10, count_dict_diet)       
coverage_array_summed = coverage_df_summed.to_numpy()
coverage_array_gene_normalized = normalize_coverage_per_gene(coverage_array_summed, gene_spacer_counts_normalized_df, 3200, 1)


# Normalize coverage for total aligned reads to make scale comparable across experiments
coverage_array_gene_and_library_size_normalized = normalize_coverage_for_tot_aligned_reads(coverage_array_gene_normalized, total_aligned_reads, 200000)
coverage_df_gene_and_library_size_normalized = pd.DataFrame(coverage_array_gene_and_library_size_normalized)
coverage_df_gene_and_library_size_normalized.columns = coverage_df_summed.columns

# Save data frame
gene_spacer_counts_normalized_df.to_csv(outdir+'gene_spacer_counts_diet.csv', index=False)
coverage_df_gene_and_library_size_normalized.to_csv(outdir+'window_coverage_data_summed_diet.csv', index=False)
with open(outdir+'tot_number_aligned_reads.txt', 'w') as file:
    # Write the integer to file
    file.write(str(total_aligned_reads))