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
        os.path.join(OUTPUTS, 'XY_data_'+str(WINDOW_SIZE)+'_'+str(WINDOW_OVERLAP)+'.npz')

        


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
        os.path.join(OUTPUTS, 'XY_data_'+str(WINDOW_SIZE)+'_'+str(WINDOW_OVERLAP)+'.npz')

    run:
        shell('python sequence_data_prep.py \
                    --operons {INPUTS}{OPERONS} \
                    --genes {INPUTS}{GENES} --outPath {OUTPUTS} --winwidth {WINDOW_SIZE} \
                    --genomelen {GENOME_LENGTH} --genome {INPUTS}{GENOME} \
                    --overlap {WINDOW_OVERLAP} --coverage_df {input}')
