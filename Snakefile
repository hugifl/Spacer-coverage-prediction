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

DATASET = str(config['dataset_name'])
WINDOW_SIZE = int(config['window_size'])
WINDOW_OVERLAP = int(config['window_overlap'])
GENOME_LENGTH = int(config['genome_length'])
MINCOUNT = int(config['mincount'])
GENE_PERC = int(config['gene_perc'])
BINSIZE = int(config['binsize'])
BATCHSIZE = int(config['bam_batchsize'])
BAMFILE_START = str(config['bamfile_start'])
READS_PER_EXPERIMENT = str(config['reads_per_experiment'])
NORMALIZATION_UNIT = str(config['normalization_unit'])
REFERENCE_GENOME = str(config['reference_genome'])

INPUTS_ANNOTATION = config['input_directory_annotation']
INPUTS_READS = config['input_directory_reads']
OUTDIR = config['output_directory']
DATA_OUTDIR = config['data_output_directory']

TUs = config['TU_file']
GENES = config['gene_file']
PROMOTERS = config['promoter_file']
TERMINATORS = config['terminator_file']
GENOME = config['genome_fasta']
COUNTS = config['count_matrix']
BAMS = config['bam_files']
BAM_FILES = glob.glob(os.path.join(INPUTS_READS, BAMS, '*.bam'))

OUTPUTS = os.path.join(OUTDIR, DATASET+"_outputs/")

if not os.path.exists(OUTPUTS):
    os.makedirs(OUTPUTS)

DATA_OUTPUTS = os.path.join(DATA_OUTDIR, DATASET+"_data/")

if not os.path.exists(OUTPUTS):
    os.makedirs(OUTPUTS)


rule all:
    input:
        os.path.join(DATA_OUTPUTS, 'window_coverage_data_summed.csv'),
        os.path.join(DATA_OUTPUTS,'XY_data_Y_with_windows.npz'), 
        os.path.join(DATA_OUTPUTS,'train_test_data_normalized_windows_info_.npz'), 
        os.path.join(DATA_OUTPUTS, 'gene_spacer_counts.csv'),
        os.path.join(DATA_OUTPUTS, 'tot_number_aligned_reads.txt')
        
rule spacer_coverage_data_prep:
    input:
        BAM_FILES = BAM_FILES
    output:
        os.path.join(DATA_OUTPUTS, 'window_coverage_data_summed.csv'),
        os.path.join(DATA_OUTPUTS, 'gene_spacer_counts.csv'),
        os.path.join(DATA_OUTPUTS, 'tot_number_aligned_reads.txt')

    run:
        shell('python spacer_coverage_data_prep.py \
                    --inbamlist {BAM_FILES} \
                    --genes {INPUTS_ANNOTATION}{GENES} --TUs {INPUTS_ANNOTATION}{TUs} --outPath {DATA_OUTPUTS} --winwidth {WINDOW_SIZE} \
                    --count_matrix {INPUTS_ANNOTATION}{COUNTS} --batchsize {BATCHSIZE} --binsize {BINSIZE} --genomelen {GENOME_LENGTH} \
                    --overlap {WINDOW_OVERLAP} --mincount {MINCOUNT} --geneperc {GENE_PERC} --bamfile_start {BAMFILE_START} --readsperexp {READS_PER_EXPERIMENT} \
                    --normalization_unit {NORMALIZATION_UNIT} --reference_genome {REFERENCE_GENOME}')


rule sequence_data_prep:
    input:
        os.path.join(DATA_OUTPUTS,'window_coverage_data_summed.csv')
    output:
        os.path.join(DATA_OUTPUTS,'XY_data_Y_with_windows.npz')

    run:
        shell('python sequence_data_prep.py \
                    --promoters {INPUTS_ANNOTATION}{PROMOTERS} --terminators {INPUTS_ANNOTATION}{TERMINATORS} \
                    --genes {INPUTS_ANNOTATION}{GENES} --TUs {INPUTS_ANNOTATION}{TUs} --outPath {DATA_OUTPUTS} --winwidth {WINDOW_SIZE} \
                    --genomelen {GENOME_LENGTH} --genome {INPUTS_ANNOTATION}{GENOME} \
                    --overlap {WINDOW_OVERLAP} --coverage_df {input}')



rule prepare_train_test_data:
    input:
        os.path.join(DATA_OUTPUTS,'XY_data_Y_with_windows.npz') 
    output:
        os.path.join(DATA_OUTPUTS,'train_test_data_normalized_windows_info_.npz') 


    run:
        shell('python prepare_train_test_data.py --counts {DATA_OUTPUTS}gene_spacer_counts.csv --outPath {DATA_OUTPUTS} --overlap {WINDOW_OVERLAP} \
                    --winwidth {WINDOW_SIZE} --binsize {BINSIZE}')
