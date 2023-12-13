### Florian Hugi ### xx-11-2023 ###


from __future__ import division
import HTSeq
import numpy
from matplotlib import pyplot
import argparse
from scipy.interpolate import interp1d
import csv
import pandas as pd
import math
from utils_coverage import filter_bamlist
from utils_coverage import total_count_per_bam
from utils_coverage import get_normalized_spacer_counts_per_gene, filter_windows, get_windows, process_batch


parser = argparse.ArgumentParser()

optional = parser._action_groups.pop()
required = parser.add_argument_group('required arguments')  # change grouping of argument to required or optional in help through this chunk of code

### inputs ###

# required 
required.add_argument('-i','--inbamlist',  nargs='+', help='array of bam files with path', required=True)
required.add_argument('-ge', '--genes', help='genome annotation file path (ECOCYC tsv)', required=True)
required.add_argument('-cou', '--count_matrix', help='matrix with gene counts per sample', required=True)
required.add_argument('-bam', '--bamfile_start', help='Pattern at the start of the bam filenames ', required=True)


# optional
optional.add_argument('-o', '--outPath', help='path to output directory.', default='.')
optional.add_argument('-w', '--winwidth', help='sequence length of input sequence, default = 1000', type=int, dest='winwidth', default=2000)
optional.add_argument('-genlen', '--genomelen', help='genome length', type=int, default=4641652)
optional.add_argument('-ov', '--overlap', help='overlap between windows', type=int, dest='overlap', default=400)
optional.add_argument('-mc', '--mincount', help='minimum amount of spacers per sample', type=int, default=10000)
optional.add_argument('-gp', '--geneperc', help='percent genes with lowest expression to be ignored', type=int, default=10)
optional.add_argument('-bs', '--binsize', help='binsize to average coverage, power of 2', type=int, default=16)
optional.add_argument('-bas', '--batchsize', help='batchsize of bamfiles to process', type=int, default=10)

#initialize values

parser._action_groups.append(optional) 
args = parser.parse_args()
bamlist = args.inbamlist
gene_file = str(args.genes)
bam_file_start = str(args.bamfile_start)
outdir=str(args.outPath)
window_size = int(args.winwidth)
overlap_size = int(args.overlap)
genome_length = int(args.genomelen)
min_counts_per_sample = int(args.mincount)
count_matrix = args.count_matrix
gene_perc = int(args.geneperc)
binsize = int(args.binsize)
batch_size = int(args.batchsize)
############ loading operon and gene information data ############

gene_df = pd.read_csv(gene_file, sep='\t')
gene_df.drop(gene_df.columns[1], axis=1, inplace=True)
gene_df.dropna(inplace=True)


count_df = pd.read_csv(count_matrix, sep='\t', dtype=str, low_memory=False)

for col in count_df.columns[5:]:
    count_df[col] = pd.to_numeric(count_df[col], errors='coerce')
    

count_dict = total_count_per_bam(count_df) # dictionary storing the total spacer counts per bam file

############ removing bam files with low spacer counts ###########
bam_directory = bamlist[0].split(bam_file_start,1)[0]
bamlist = [bam_file_start + s.split(bam_file_start, 1)[1] if bam_file_start in s else s for s in bamlist] 


bamlist = filter_bamlist(bamlist, count_df, min_counts_per_sample)
# Compute the windows
windows = get_windows(genome_length,window_size,overlap_size)

batch = 0
for i in range(0, len(bamlist), batch_size):
    batch += 1
    print("batch " + str(batch) + " out of "+ str(len(bamlist)/batch_size))
    bam_batch = bamlist[i:i + batch_size]
    batch_coverage_df = process_batch(bam_batch, windows, count_dict, binsize, bam_directory)
    batch_coverage_summed = batch_coverage_df.groupby(['Window_Start', 'Window_End']).sum().reset_index()
    
    if batch == 1:
        coverage_df_summed = batch_coverage_summed
    else:
        # Sum the batch coverage with the total coverage
        coverage_df_summed = pd.concat([coverage_df_summed, batch_coverage_summed]).groupby(['Window_Start', 'Window_End']).sum().reset_index()

# Filter out windows that contain low-expressed genes
low_expressed_genes, gene_spacer_counts_normalized_df = get_normalized_spacer_counts_per_gene(count_df, gene_perc, count_dict)       
coverage_df_summed = filter_windows(coverage_df_summed, low_expressed_genes, gene_df)   

# Save data frame
gene_spacer_counts_normalized_df.to_csv(outdir+'gene_spacer_counts.csv', index=False)
coverage_df_summed.to_csv(outdir+'window_coverage_data_summed.csv', index=False)



