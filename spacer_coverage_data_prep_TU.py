### Florian Hugi ### xx-03-2024 ###


from __future__ import division
import HTSeq
import numpy
from matplotlib import pyplot
import argparse
from scipy.interpolate import interp1d
import csv
import pandas as pd
import math
from utils_coverage import filter_bamlist, expand_count_df, replace_summed_pads
from utils_coverage import total_count_per_bam, remove_lowest_expressed_TUs, normalize_coverage_for_tot_aligned_reads_TU
from utils_coverage import get_normalized_spacer_counts_per_gene, filter_windows, get_windows, process_batch_TU, get_normalized_spacer_counts_per_TU, normalize_TU_coverage_per_TU
import os

parser = argparse.ArgumentParser()

optional = parser._action_groups.pop()
required = parser.add_argument_group('required arguments')  # change grouping of argument to required or optional in help through this chunk of code

### inputs ###

# required 
required.add_argument('-i','--inbamlist',  nargs='+', help='array of bam files with path', required=True)
required.add_argument('-ge', '--genes', help='genome annotation file path (ECOCYC tsv)', required=True)
required.add_argument('-tu', '--TUs', help='genome annotation file path (ECOCYC tsv)', required=True)
required.add_argument('-cou', '--count_matrix', help='matrix with gene counts per sample', required=True)
required.add_argument('-bam', '--bamfile_start', help='Pattern at the start of the bam filenames ', required=True)


# optional
optional.add_argument('-o', '--outPath', help='path to output directory.', default='.')
optional.add_argument('-genlen', '--genomelen', help='genome length', type=int, default=4641652)
optional.add_argument('-mc', '--mincount', help='minimum amount of spacers per sample', type=int, default=10000)
optional.add_argument('-gp', '--geneperc', help='percent genes with lowest expression to be ignored', type=int, default=10)
optional.add_argument('-bs', '--binsize', help='binsize to average coverage, power of 2', type=int, default=16)
optional.add_argument('-bas', '--batchsize', help='batchsize of bamfiles to process', type=int, default=10)
optional.add_argument('-exp', '--readsperexp', help='expected number of aligned reads per experiment (across all samples)', type=int, default=10000000)
optional.add_argument('-no', '--normalization_unit', help='normalization unit', type=str, default='gene')
optional.add_argument('-ref', '--reference_genome', help='reference genome name', type=str, default='gene') 
optional.add_argument('-pad', '--padsymbol', help='Symbol to pad TUs', type=str, default='N') 

#initialize values

parser._action_groups.append(optional) 
args = parser.parse_args()
bamlist = args.inbamlist
gene_file = str(args.genes)
TU_file = str(args.TUs)
bam_file_start = str(args.bamfile_start)
outdir=str(args.outPath)
genome_length = int(args.genomelen)
min_counts_per_sample = int(args.mincount)
count_matrix = args.count_matrix
gene_perc = int(args.geneperc)
binsize = int(args.binsize)
batch_size = int(args.batchsize)
expected_aligned_reads_per_experiment = int(args.readsperexp)
normalization_unit = str(args.normalization_unit)
reference_genome = str(args.reference_genome)
pad_symbol = str(args.padsymbol)
############ loading operon and gene information data ############

gene_df = pd.read_csv(gene_file, sep='\t')
gene_df.drop(gene_df.columns[1], axis=1, inplace=True)
gene_df.dropna(inplace=True)

TU_df = pd.read_csv(TU_file, sep='\t')
TU_df.dropna(inplace=True)


count_df = pd.read_csv(count_matrix, sep='\t', dtype=str, low_memory=False)

for col in count_df.columns[5:]:
    count_df[col] = pd.to_numeric(count_df[col], errors='coerce')
    
count_dict = total_count_per_bam(count_df, bam_file_start) # dictionary storing the total spacer counts per bam file
print("example key-value pair of count_dict: " + str(list(count_dict.items())[0]))
############ removing bam files with low spacer counts ###########
bam_directory = bamlist[0].split(bam_file_start,1)[0]
bamlist = [bam_file_start + s.split(bam_file_start, 1)[1] if bam_file_start in s else s for s in bamlist] 
print(len(bamlist))

bamlist, count_df = filter_bamlist(bamlist, count_df, min_counts_per_sample, bam_file_start) 
print(len(bamlist))
print("example bam file name: " + str(bamlist[0]))

count_dict = {k: count_dict[k] for k in bamlist if k in count_dict}

if len(count_dict) == 0:
    raise RuntimeError("The dictionary is empty.")
# Assuming the DataFrame is stored as a CSV file in the 'out' directory
coverage_df_path = outdir+'coverage_df_summed.csv'

# Check if the file exists
if os.path.exists(coverage_df_path):
    print("Loading coverage_df_summed from file")
    # If the file exists, load the DataFrame
    coverage_df_summed = pd.read_csv(coverage_df_path)
    total_aligned_reads_file = outdir + 'total_aligned_reads.txt'
    with open(total_aligned_reads_file, "r") as file:
        total_aligned_reads = int(file.read())
else:
    batch = 0
    total_aligned_reads = 0
    total_window_aligned_reads = 0
    total_reads = 0
    for i in range(0, len(bamlist), batch_size):
        batch += 1
        print("batch " + str(batch) + " out of "+ str(len(bamlist)/batch_size))
        bam_batch = bamlist[i:i + batch_size]
        batch_coverage_df, aligned_read_count_batch, window_aligned_read_count_batch, total_read_count_batch = process_batch_TU(bam_batch, count_df, reference_genome, bam_directory, pad_symbol)
        total_aligned_reads += aligned_read_count_batch
        total_window_aligned_reads += window_aligned_read_count_batch
        total_reads += total_read_count_batch
        batch_coverage_summed = batch_coverage_df.groupby(['TU_Name', 'Start', 'End', 'Direction']).sum().reset_index()
        print("head of batch_coverage_summed", batch_coverage_summed.head())

        if batch == 1:
            coverage_df_summed = batch_coverage_summed
        else:
            # Sum the batch coverage with the total coverage
            coverage_df_summed = pd.concat([coverage_df_summed, batch_coverage_summed]).groupby(['TU_Name', 'Start', 'End', 'Direction']).sum().reset_index()
    
    coverage_df_path_2 = outdir+'coverage_df_summed_raw.csv'
    coverage_df_summed.to_csv(coverage_df_path_2, index=False)
    coverage_df_summed = replace_summed_pads(coverage_df_summed, pad_symbol)
    coverage_df_summed.to_csv(coverage_df_path, index=False)

    count_stats_file = outdir + 'count_stats.txt'
    total_aligned_reads_file = outdir + 'tot_number_aligned_reads.txt'
    with open(count_stats_file, 'w') as file:
        # Write descriptive text and the integer value
        file.write('Total reads: ' + str(total_reads) + '\n')
        file.write('Total aligned reads: ' + str(total_aligned_reads) + '\n')
        file.write('Total reads aligned to windows: ' + str(total_window_aligned_reads) + '\n')
    with open(total_aligned_reads_file, "w") as file:
        file.write(str(total_aligned_reads))


# Sometimes genes that are on insertion elements will have multiple start and end sites in the count matrix.
count_df = expand_count_df(count_df)
print("count dataframe expanded")


coverage_df_TU_normalized_path = outdir+'coverage_df_TU_normalized.csv'
if os.path.exists(coverage_df_TU_normalized_path):
    print("Loading expression normalized coverage from file")
    coverage_df_TU_normalized = pd.read_csv(coverage_df_TU_normalized_path)
else:
    # Normalize coverage for gene expression (RPKM) values per gene to remove effects of differential gene expression.
    print("count df rownumber: " + str(count_df.shape[0]) )
    low_expressed_TUs, TU_spacer_counts_normalized_df = get_normalized_spacer_counts_per_TU(count_df, gene_perc, count_dict) 
    file_path = outdir + 'TU_spacer_counts.csv'
    test_dir = outdir + 'test.csv'
    test_dir_2 = outdir + 'spacer_counts_TU.csv'
    test_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    print("saving TU_spacer_counts_normalized_df to: " +file_path)
    TU_spacer_counts_normalized_df.to_csv(file_path, index=False)
    test_df_2 = TU_spacer_counts_normalized_df
    test_df.to_csv(test_dir, index=False)
    test_df_2.to_csv(test_dir_2, index=False)

    if os.path.exists(file_path):
        print("File successfully saved at:", file_path)
    else:
        print("Failed to save the file.")

    try:
        df = pd.read_csv(file_path)
        print(df.head())
    except FileNotFoundError as e:
        print(e)
    if os.path.exists(test_dir):
        print("File successfully saved at:", test_dir)
    else:
        print("Failed to save the file.")

    try:
        df = pd.read_csv(test_dir)
        print(df.head())
    except FileNotFoundError as e:
        print(e)
        
    #coverage_df_summed = remove_lowest_expressed_TUs(coverage_df_summed, low_expressed_TUs)        
    coverage_df_TU_normalized = normalize_TU_coverage_per_TU(coverage_df_summed, TU_spacer_counts_normalized_df, pad_symbol)
    coverage_df_TU_normalized.to_csv(coverage_df_TU_normalized_path, index=False)
    print("normalize for total aligned reads to make scale comparable across experiments")
    # Normalize coverage for total aligned reads to make scale comparable across experiments

coverage_df_TU_and_library_size_normalized = normalize_coverage_for_tot_aligned_reads_TU(coverage_df_TU_normalized, total_aligned_reads, expected_aligned_reads_per_experiment)
coverage_df_TU_and_library_size_normalized_lowest_TU_removed = remove_lowest_expressed_TUs(coverage_df_TU_and_library_size_normalized, low_expressed_TUs)    
print("Saving data")
# save low expressed TUs
low_expressed_TUs_file = outdir + 'low_expressed_TUs.csv'
low_expressed_TUs_df = pd.DataFrame(low_expressed_TUs)
low_expressed_TUs_df.to_csv(low_expressed_TUs_file, index=False)

coverage_df_TU_and_library_size_normalized.to_csv(outdir+'TU_coverage_data_summed.csv', index=False)
coverage_df_TU_and_library_size_normalized_lowest_TU_removed.to_csv(outdir+'TU_coverage_data_summed_TU_removed.csv', index=False)
with open(outdir+'tot_number_aligned_reads.txt', 'w') as file:
    # Write the integer to file
    file.write(str(total_aligned_reads))




