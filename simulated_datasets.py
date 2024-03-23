from __future__ import division
import numpy as np
import numpy
from matplotlib import pyplot
import argparse
from scipy.interpolate import interp1d
import csv
import pandas as pd
import math
from utils_simulated_data import generate_random_dna_sequences, generate_annotation_features, generate_dummy_sequences, generate_annotation_features_2_genes
import os

outdir = '/cluster/scratch/hugifl/spacer_coverage_final_data_2/Simulated_datasets'
os.makedirs(outdir, exist_ok=True)

num_sequences = 100
seq_length = 3200


## ------------------------------------------------ Fully Random sequence gene center ------------------------------------------------
#name = 'fully_random_gene_center'
#outpath = outdir + '/' + name
#
#
#gc_content = 0.5
#len_gene = 1000
#nu_prom = 20
#gene_direction = '+'
#nu_terminator = 5
#terminator_length = 20
#
#X_test_seq = generate_random_dna_sequences(num_sequences, seq_length, gc_content)
#X_test_anno = generate_annotation_features(num_sequences, seq_length, len_gene, nu_prom, gene_direction, nu_terminator, terminator_length, gene_position = 'center')
#print("shape of X_test_seq", X_test_seq.shape)
#print("shape of X_test_anno", X_test_anno.shape)
#X_test = np.concatenate((X_test_seq, X_test_anno), axis=2)
#
#numpy.savez(outpath+'.npz', X=X_test)
#
#
## ------------------------------------------------ Fully Random sequence gene center GC 0.7 ------------------------------------------------
#name = 'fully_random_gene_center_gc_0.7'
#outpath = outdir + '/' + name
#
#
#gc_content = 0.7
#len_gene = 1000
#nu_prom = 20
#gene_direction = '+'
#nu_terminator = 5
#terminator_length = 20
#
#X_test_seq = generate_random_dna_sequences(num_sequences, seq_length, gc_content)
#X_test_anno = generate_annotation_features(num_sequences, seq_length, len_gene, nu_prom, gene_direction, nu_terminator, terminator_length, gene_position = 'center')
#print("shape of X_test_seq", X_test_seq.shape)
#print("shape of X_test_anno", X_test_anno.shape)
#X_test = np.concatenate((X_test_seq, X_test_anno), axis=2)
#
#numpy.savez(outpath+'.npz', X=X_test)
#
## ------------------------------------------------ Fully Random sequence gene center GC 0.3 ------------------------------------------------
#name = 'fully_random_gene_center_gc_0.3'
#outpath = outdir + '/' + name
#
#
#gc_content = 0.3
#len_gene = 1000
#nu_prom = 20
#gene_direction = '+'
#nu_terminator = 5
#terminator_length = 20
#
#X_test_seq = generate_random_dna_sequences(num_sequences, seq_length, gc_content)
#X_test_anno = generate_annotation_features(num_sequences, seq_length, len_gene, nu_prom, gene_direction, nu_terminator, terminator_length, gene_position = 'center')
#print("shape of X_test_seq", X_test_seq.shape)
#print("shape of X_test_anno", X_test_anno.shape)
#X_test = np.concatenate((X_test_seq, X_test_anno), axis=2)
#
#numpy.savez(outpath+'.npz', X=X_test)

 
# ------------------------------------------------ Dummy sequence gene center ------------------------------------------------
#name = 'dummy_ATCG_gene_center'
#outpath = outdir + '/' + name
#
#dummy_sequence = 'ATCG'
#len_gene = 1000
#nu_prom = 60
#gene_direction = '+'
#nu_terminator = 5
#terminator_length = 30
#
#X_test_seq = generate_dummy_sequences(num_sequences, seq_length, dummy_sequence)
#X_test_anno = generate_annotation_features(num_sequences, seq_length, len_gene, nu_prom, gene_direction, nu_terminator, terminator_length, gene_position = 'center')
#print("shape of X_test_seq", X_test_seq.shape)
#print("shape of X_test_anno", X_test_anno.shape)
#X_test = np.concatenate((X_test_seq, X_test_anno), axis=2)
#
#numpy.savez(outpath+'.npz', X=X_test)

# ------------------------------------------------ Dummy sequence gene start ------------------------------------------------
#name = 'dummy_ATCG_gene_end'
#outpath = outdir + '/' + name
#
#dummy_sequence = 'ATCG'
#len_gene = 1000
#nu_prom = 60
#gene_direction = '+'
#nu_terminator = 5
#terminator_length = 30
#
#X_test_seq = generate_dummy_sequences(num_sequences, seq_length, dummy_sequence)
#X_test_anno = generate_annotation_features(num_sequences, seq_length, len_gene, nu_prom, gene_direction, nu_terminator, terminator_length, gene_position = 'end')
#print("shape of X_test_seq", X_test_seq.shape)
#print("shape of X_test_anno", X_test_anno.shape)
#X_test = np.concatenate((X_test_seq, X_test_anno), axis=2)
#
#numpy.savez(outpath+'.npz', X=X_test)

# ------------------------------------------------ Dummy sequence gene center reverse ------------------------------------------------
#name = 'dummy_ATCG_gene_center_reverse'
#outpath = outdir + '/' + name
#
#dummy_sequence = 'ATCG'
#len_gene = 1000
#nu_prom = 60
#gene_direction = '-'
#nu_terminator = 5
#terminator_length = 30
#
#X_test_seq = generate_dummy_sequences(num_sequences, seq_length, dummy_sequence)
#X_test_anno = generate_annotation_features(num_sequences, seq_length, len_gene, nu_prom, gene_direction, nu_terminator, terminator_length, gene_position = 'center')
#print("shape of X_test_seq", X_test_seq.shape)
#print("shape of X_test_anno", X_test_anno.shape)
#X_test = np.concatenate((X_test_seq, X_test_anno), axis=2)
#
#numpy.savez(outpath+'.npz', X=X_test)

## ------------------------------------------------ Dummy sequence gene center AT rich islets ------------------------------------------------
#name = 'dummy_ATCG_gene_center_AG_islets'
#outpath = outdir + '/' + name
#
#dummy_sequence = 'ATCG'
#dummy_sequence_2 = 'ACGGAGTAGATCAGAGATAGACCATACCTAGAGGAAG'
#len_gene = 1000
#nu_prom = 60
#gene_direction = '+'
#nu_terminator = 5
#terminator_length = 30
#
#X_test_seq = generate_dummy_sequences(num_sequences, seq_length, dummy_sequence, dummy_sequence_2)
#X_test_anno = generate_annotation_features(num_sequences, seq_length, len_gene, nu_prom, gene_direction, nu_terminator, terminator_length, gene_position = 'center')
#print("shape of X_test_seq", X_test_seq.shape)
#print("shape of X_test_anno", X_test_anno.shape)
#X_test = np.concatenate((X_test_seq, X_test_anno), axis=2)
#
#numpy.savez(outpath+'.npz', X=X_test)


# ------------------------------------------------ Dummy sequence two gene AT rich islets ------------------------------------------------
name = 'dummy_ATCG_two_genes_AG_islets'
outpath = outdir + '/' + name

dummy_sequence = 'ATCG'
dummy_sequence_2 = 'ACGGAGTAGATCAGAGATAGACCATACCTAGAGGAAG'
len_gene = 1000
nu_prom = 60
gene_direction = '+'
nu_terminator = 5
terminator_length = 30

X_test_seq = generate_dummy_sequences(num_sequences, seq_length, dummy_sequence, dummy_sequence_2)
X_test_anno = generate_annotation_features_2_genes(num_sequences, seq_length, len_gene, nu_prom, gene_direction, nu_terminator, terminator_length)
print("shape of X_test_seq", X_test_seq.shape)
print("shape of X_test_anno", X_test_anno.shape)
X_test = np.concatenate((X_test_seq, X_test_anno), axis=2)

numpy.savez(outpath+'.npz', X=X_test)

