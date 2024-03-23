### Florian Hugi ### xx-11-2023 ###


from __future__ import division
import numpy
import argparse
import pandas as pd
from utils_sequence import parse_fasta, one_hot_encode, extract_sequences_and_sequence_info_TU, dataframe_to_2darray_keep_window_information

parser = argparse.ArgumentParser()

optional = parser._action_groups.pop()
required = parser.add_argument_group('required arguments')  # change grouping of argument to required or optional in help through this chunk of code

### inputs ###

# required 
required.add_argument('-pr', '--promoters', help='genome annotation file path (ECOCYC tsv)', required=True)
required.add_argument('-te', '--terminators', help='genome annotation file path (ECOCYC tsv)', required=True)
required.add_argument('-ge', '--genes', help='genome annotation file path (regulonDB tsv)', required=True)
required.add_argument('-gen', '--genome', help='genome sequence, raw sequence', required=True)
required.add_argument('-cdf', '--coverage_df', help='coverage dataframe', required = True)
required.add_argument('-tu', '--TUs', help='genome annotation file path (ECOCYC tsv)', required=True)

# optional
optional.add_argument('-o', '--outPath', help='path to output directory.', default='.')
optional.add_argument('-w', '--winwidth', help='sequence length of input sequence, default = 1000', type=int, dest='winwidth', default=2000)
optional.add_argument('-genlen', '--genomelen', help='genome length', type=int, default=4641652)
optional.add_argument('-ov', '--overlap', help='overlap between windows', type=int, dest='overlap', default=400)
optional.add_argument('-pad', '--padsymbol', help='Symbol to pad TUs', type=str, default='N') 

#initialize values

parser._action_groups.append(optional) 
args = parser.parse_args()

pad_symbol = str(args.padsymbol)
TU_file = str(args.TUs)
promoter_file = str(args.promoters)
terminator_file = str(args.terminators)
gene_file = str(args.genes)
genome_file = str(args.genome)
outdir=str(args.outPath)
window_size = int(args.winwidth)
genome_length = int(args.genomelen)
coverage_df_file = str(args.coverage_df)
overlap_size = int(args.overlap)

############ loading operon and gene information data ############

promoter_df = pd.read_csv(promoter_file, sep='\t')
promoter_df.dropna(inplace=True)

terminator_df = pd.read_csv(terminator_file, sep='\t')
terminator_df.dropna(inplace=True)


gene_df = pd.read_csv(gene_file, sep='\t')
gene_df.drop(gene_df.columns[1], axis=1, inplace=True)
gene_df.dropna(inplace=True)

TU_df = pd.read_csv(TU_file, sep='\t')
TU_df.dropna(inplace=True)         
    
# Load coverage dataframe (already filtered for low expressed genes and summed over all samples)
coverage_df_summed = pd.read_csv(coverage_df_file, sep=',', comment="#")

# keep only the window information and the coverage data
selected_columns = ['Start', 'End'] + [col for col in coverage_df_summed.columns if col.startswith('Pos_')]

# Keep only the selected columns in the DataFrame
coverage_df_summed_reduced = coverage_df_summed[selected_columns]

# turn coverage data into shape that can be used in ML models
Y = dataframe_to_2darray_keep_window_information(coverage_df_summed_reduced) 

# produce input data X (DNA sequence, gene and operon start/end sites, gene direction)
genome = parse_fasta(genome_file)
X = extract_sequences_and_sequence_info_TU(coverage_df_summed, genome, gene_df, promoter_df, terminator_df, TU_df, pad_symbol)
#numpy.savez(outdir+'XY_data_'+str(window_size)+'_'+str(overlap_size)+'.npz', X=X, Y=Y)

numpy.savez(outdir+'XY_data_Y_with_windows.npz', X=X, Y=Y)