from snakemake.utils import R
import glob
import os
import logging
import json
import fnmatch
import re
from collections import defaultdict
import csv

# Globals ---------------------------------------------------------------------

configfile: "config.yml"

WINDOW_SIZE = int(config['window_size'])
WINDOW_OVERLAP = int(config['window_overlap'])
GENOME_LENGTH = int(config['genome_length'])
MINCOUNT = int(config['mincount'])
GENE_PERC = int(config['gene_perc'])
BINSIZE = int(config['binsize'])
BATCHSIZE = int(config['bam_batchsize'])
NORMALIZATION_UNIT = str(config['normalization_unit'])

INPUTS = config['input_directory']
OUTPUTS = config['output_directory']

GENES = config['gene_tsv']
OPERONS = config['operon_tsv']
GENOME = config['genome_fasta']
COUNTS = config['count_matrix']
BAMS = config['bam_files']
BAM_FILES = glob.glob(os.path.join(INPUTS, BAMS, '*.bam'))



rule all:
    input:
        os.path.join(OUTPUTS, 'window_coverage_data_summed_' + str(WINDOW_SIZE) + '_' + str(WINDOW_OVERLAP) + '.csv'),
        os.path.join(OUTPUTS,'XY_data_Y_with_windows'+str(WINDOW_SIZE)+'_'+str(WINDOW_OVERLAP)+'.npz'), #  os.path.join(OUTPUTS, 'XY_data_'+str(WINDOW_SIZE)+'_'+str(WINDOW_OVERLAP)+'.npz'),
        os.path.join(OUTPUTS,'train_test_data_binary_windows_info_' + str(WINDOW_SIZE)+'_'+str(WINDOW_OVERLAP) + '.npz'),
        os.path.join(OUTPUTS,'train_test_data_binary_' + str(WINDOW_SIZE)+'_'+str(WINDOW_OVERLAP) + '.npz'),
        os.path.join(OUTPUTS, NORMALIZATION_UNIT  + '_spacer_counts.csv')
        
rule coverage_data_prep:
    input:
        BAM_FILES = BAM_FILES
    output:
        os.path.join(OUTPUTS, 'window_coverage_data_summed_' + str(WINDOW_SIZE) + '_' + str(WINDOW_OVERLAP) + '.csv')

    run:
        shell('python spacer_coverage_data_prep.py \
                    --inbamlist {BAM_FILES} --operons {INPUTS}{OPERONS} \
                    --genes {INPUTS}{GENES} --outPath {OUTPUTS} --winwidth {WINDOW_SIZE} \
                    --count_matrix {INPUTS}{COUNTS} --batchsize {BATCHSIZE} --binsize {BINSIZE} --genomelen {GENOME_LENGTH} \
                    --overlap {WINDOW_OVERLAP} --mincount {MINCOUNT} --geneperc {GENE_PERC}')


rule sequence_data_prep:
    input:
        os.path.join(OUTPUTS, 'window_coverage_data_summed_' + str(WINDOW_SIZE) + '_' + str(WINDOW_OVERLAP) + '.csv')
    output:
        os.path.join(OUTPUTS,'XY_data_Y_with_windows'+str(WINDOW_SIZE)+'_'+str(WINDOW_OVERLAP)+'.npz') #os.path.join(OUTPUTS, 'XY_data_'+str(WINDOW_SIZE)+'_'+str(WINDOW_OVERLAP)+'.npz'),

    run:
        shell('python sequence_data_prep.py \
                    --operons {INPUTS}{OPERONS} \
                    --genes {INPUTS}{GENES} --outPath {OUTPUTS} --winwidth {WINDOW_SIZE} \
                    --genomelen {GENOME_LENGTH} --genome {INPUTS}{GENOME} \
                    --overlap {WINDOW_OVERLAP} --coverage_df {input}')


rule spacer_count_per_normalization_unit:
    output:
        os.path.join(OUTPUTS, NORMALIZATION_UNIT  + '_spacer_counts.csv')


    run:
        shell('python spacer_count_per_normalization_unit.py --operons {INPUTS}{OPERONS} --outPath {OUTPUTS} \
                    --genes {INPUTS}{GENES} --count_matrix {INPUTS}{COUNTS} --normalization_unit {NORMALIZATION_UNIT}')

rule prepare_train_test_data:
    input:
        os.path.join(OUTPUTS,'XY_data_Y_with_windows'+str(WINDOW_SIZE)+'_'+str(WINDOW_OVERLAP)+'.npz') #os.path.join(OUTPUTS, 'XY_data_'+str(WINDOW_SIZE)+'_'+str(WINDOW_OVERLAP)+'.npz'),
    output:
        os.path.join(OUTPUTS,'train_test_data_binary_windows_info_' + str(WINDOW_SIZE)+'_'+str(WINDOW_OVERLAP) + '.npz'),
        os.path.join(OUTPUTS,'train_test_data_binary_' + str(WINDOW_SIZE)+'_'+str(WINDOW_OVERLAP) + '.npz'),
        os.path.join(OUTPUTS,'train_test_data_normalized_windows_info_' + str(WINDOW_SIZE)+'_'+str(WINDOW_OVERLAP) + '.npz'),
        os.path.join(OUTPUTS,'train_test_data_normalized_' + str(WINDOW_SIZE)+'_'+str(WINDOW_OVERLAP) + '.npz')


    run:
        shell('python prepare_train_test_data.py --counts {OUTPUTS}{NORMALIZATION_UNIT}_spacer_counts.csv --outPath {OUTPUTS} --overlap {WINDOW_OVERLAP} \
                    --winwidth {WINDOW_SIZE} --binsize {BINSIZE} --normalization_unit {NORMALIZATION_UNIT}')
