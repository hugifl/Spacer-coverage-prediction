from __future__ import division
import HTSeq
import numpy
from matplotlib import pyplot
import argparse
from scipy.interpolate import interp1d
import csv
import pandas as pd
import math
from plot_genes import plot_operons
from data_loading import data_loading
from utils_coverage import get_operon_spacer_counts, get_gene_spacer_counts


parser = argparse.ArgumentParser()

optional = parser._action_groups.pop()
required = parser.add_argument_group('required arguments')  # change grouping of argument to required or optional in help through this chunk of code

### inputs ###

# required 
required.add_argument('-op', '--operons', help='genome annotation file path (regulonDB tsv)', required=True)
required.add_argument('-ge', '--genes', help='genome annotation file path (regulonDB tsv)', required=True)
required.add_argument('-cou', '--count_matrix', help='matrix with gene counts per sample', required=True)
required.add_argument('-nu', '--normalization_unit', help='operon or gene to normalize coverage profiles per operon or per gene', required=True)


# optional
optional.add_argument('-o', '--outPath', help='path to output directory.', default='.')


#initialize values

parser._action_groups.append(optional) 
args = parser.parse_args()

tsv_file = str(args.operons)
tsv_file_2 = str(args.genes)
count_file = str(args.count_matrix)
normalization_unit = str(args.normalization_unit)
outdir = str(args.outPath)

operon_df = data_loading(tsv_file)
gene_df = data_loading(tsv_file_2)
count_df = pd.read_csv(count_file, sep='\t', dtype=str, low_memory=False)

if normalization_unit == 'operon':
    operon_counts = get_operon_spacer_counts(count_df, gene_df, operon_df)
    operon_counts.to_csv(outdir+'operon_spacer_counts.csv', index=False)
elif normalization_unit == 'gene':
    gene_spacer_counts_normalized_df = get_gene_spacer_counts(count_df, gene_df)
    gene_spacer_counts_normalized_df.to_csv(outdir+'gene_spacer_counts.csv', index=False)


operon_counts.to_csv(outdir+ normalization_unit +'_spacer_counts.csv', index=False)