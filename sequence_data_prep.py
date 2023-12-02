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
from utils_data_loading import data_loading
from utils_coverage import filter_bamlist
from utils_coverage import total_count_per_bam, bin_coverage, dataframe_to_2darray, dataframe_to_2darray_keep_window_information
from utils_coverage import lowest_expressed_genes, filter_windows, get_windows
from utils_sequence import parse_fasta, one_hot_encode, extract_sequences_and_sequence_info

parser = argparse.ArgumentParser()

optional = parser._action_groups.pop()
required = parser.add_argument_group('required arguments')  # change grouping of argument to required or optional in help through this chunk of code

### inputs ###

# required 
required.add_argument('-op', '--operons', help='genome annotation file path (regulonDB tsv)', required=True)
required.add_argument('-ge', '--genes', help='genome annotation file path (regulonDB tsv)', required=True)
required.add_argument('-gen', '--genome', help='genome sequence, raw sequence', required=True)
required.add_argument('-cdf', '--coverage_df', help='coverage dataframe', required = True)
# optional
optional.add_argument('-o', '--outPath', help='path to output directory.', default='.')
optional.add_argument('-w', '--winwidth', help='sequence length of input sequence, default = 1000', type=int, dest='winwidth', default=2000)
optional.add_argument('-genlen', '--genomelen', help='genome length', type=int, default=4641652)
optional.add_argument('-ov', '--overlap', help='overlap between windows', type=int, dest='overlap', default=400)

#initialize values

parser._action_groups.append(optional) 
args = parser.parse_args()

tsv_file = str(args.operons)
tsv_file_2 = str(args.genes)
genome_file = str(args.genome)
outdir=str(args.outPath)
window_size = int(args.winwidth)
genome_length = int(args.genomelen)
coverage_df_file = str(args.coverage_df)
overlap_size = int(args.overlap)

############ loading operon and gene information data ############

operon_dataframe = data_loading(tsv_file)
gene_dataframe = data_loading(tsv_file_2)
    
# Load coverage dataframe (already filtered and summed)
coverage_df_summed = pd.read_csv(coverage_df_file, sep=',', comment="#")

# turn coverage data into shape that can be used in ML models
#Y = dataframe_to_2darray(coverage_df_summed) 
Y_window = dataframe_to_2darray_keep_window_information(coverage_df_summed) 

# produce input data X (DNA sequence, gene and operon start/end sites, gene direction)
genome = parse_fasta(genome_file)
X = extract_sequences_and_sequence_info(coverage_df_summed, genome, window_size, operon_dataframe, gene_dataframe)

#numpy.savez(outdir+'XY_data_'+str(window_size)+'_'+str(overlap_size)+'.npz', X=X, Y=Y)

numpy.savez(outdir+'XY_data_Y_with_windows'+str(window_size)+'_'+str(overlap_size)+'.npz', X=X, Y=Y_window)