import numpy
import pandas as pd

def parse_fasta(fasta_file):
    with open(fasta_file, 'r') as file:
        genome = ''
        for line in file:
            if line.startswith('>'):
                continue
            genome += line.strip()
    return genome

def one_hot_encode(seq):
    # Map nucleotides to integers: A:0, C:1, G:2, T:3
    mapping = {'A': 0, 'G': 1, 'C': 2, 'T': 3}
    one_hot = numpy.zeros((len(seq), 4), dtype=int)
    for i, nucleotide in enumerate(seq):
        if nucleotide in mapping:
            one_hot[i, mapping[nucleotide]] = 1
    return one_hot

def extract_sequences_and_sequence_info(df, genome, window_size, operon_df, gene_df):
    print("start building sequence dataset")
    sequences = []

    no_rows = df.shape[0]
    counter = 0
    for _, row in df.iterrows():
        counter += 1
        print("fraction done: ")
        print(counter / no_rows)
        window_start, window_end = int(row['Window_Start']), int(row['Window_End'])
        seq = genome[window_start:window_end]
        if len(seq) != window_size:  
            print("Window size doesn't match effective size of windows")
            continue

        # One-hot encode DNA sequence
        encoded_seq = one_hot_encode(seq)
        # Initialize binary vectors
        gene_vector = numpy.zeros(window_size, dtype=int)
        operon_vector = numpy.zeros(window_size, dtype=int)
        operon_directionality_vector = numpy.zeros(window_size, dtype=int)

        # Populate gene vector
        for _, gene_row in gene_df.iterrows():

            if gene_row['leftEndPos'] == 'None' or gene_row['rightEndPos']  == 'None':
                continue
            
            gene_start = int(gene_row['leftEndPos']) - window_start -1 # -1 because genome sequenced is indexed from 0 but gene and operon locations are indexed from 1
            gene_end = int(gene_row['rightEndPos']) - window_start -1 
            if 0 <= gene_start < window_size:
                gene_vector[gene_start] = 1
            if 0 <= gene_end < window_size:
                gene_vector[gene_end] = 1

        # Populate operon and directionality vectors
        for _, operon in operon_df.iterrows():
            operon_start = int(operon['firstGeneLeftPos']) - window_start -1 
            operon_end = int(operon['lastGeneRightPos']) - window_start -1 
            operon_strand = 1 if operon['strand'] == 'forward' else -1

            # Ensure correct ordering of start and end
            operon_start, operon_end = min(operon_start, operon_end), max(operon_start, operon_end)

            # start and end sites of operons are marked
            if (0 <= operon_start < window_size):
                operon_vector[operon_start] = 1

            if (0 <= operon_end < window_size):
                operon_vector[operon_end] = 1

            # check if any part of the operon is within the window
            if (0 <= operon_start < window_size) or (0 <= operon_end < window_size) or (operon_start < 0 and operon_end >= window_size):

                operon_start = max(0, operon_start)
                operon_end = min(window_size - 1, operon_end)

                # Operon body is 1 or -1 depending on directionality (0 if no operon present)
                operon_directionality_vector[operon_start:operon_end + 1] = operon_strand


        # Concatenate all vectors
        full_vector = numpy.concatenate((encoded_seq, gene_vector[:, None], operon_vector[:, None], operon_directionality_vector[:, None]), axis=1)
        sequences.append(full_vector)

    return numpy.array(sequences)