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

def dataframe_to_2darray_keep_window_information(df):
    coverage_array = df.to_numpy()

    return coverage_array

def one_hot_encode(seq):
    # Map nucleotides to integers: A:0, C:1, G:2, T:3
    mapping = {'A': 0, 'G': 1, 'C': 2, 'T': 3}
    one_hot = numpy.zeros((len(seq), 4), dtype=int)
    for i, nucleotide in enumerate(seq):
        if nucleotide in mapping:
            one_hot[i, mapping[nucleotide]] = 1
    return one_hot

def extract_sequences_and_sequence_info(df, genome, window_size, gene_df, promoter_df, terminator_df):
    print("start building sequence dataset")
    sequences = []

    no_rows = df.shape[0]
    counter = 0
    for _, row in df.iterrows():
        counter += 1
        #print("fraction done: ")
        #print(counter / no_rows)
        window_start, window_end = int(row['Window_Start']), int(row['Window_End'])
        seq = genome[window_start:window_end]

        # One-hot encode DNA sequence
        encoded_seq = one_hot_encode(seq)
        # Initialize binary vectors
        gene_vector = numpy.zeros(window_size, dtype=int)
        promoter_vector = numpy.zeros(window_size, dtype=int)
        terminator_vector = numpy.zeros(window_size, dtype=int)
        gene_directionality_vector = numpy.zeros(window_size, dtype=int)

        # Populate gene vector
        for _, gene_row in gene_df.iterrows():

            if gene_row['Left'] == 'None' or gene_row['Right']  == 'None':
                continue
            #print("gene row: ")
            #print(gene_row)
            #print("gene row left: ")
            #print(gene_row['Left'])
            #print("gene row right: ")
            #print(gene_row['Right'])
            gene_start = int(gene_row['Left']) - window_start 
            gene_end = int(gene_row['Right']) - window_start 
            if 0 <= gene_start < window_size:
                gene_vector[gene_start] = 1
            if 0 <= gene_end < window_size:
                gene_vector[gene_end] = 1
            
            gene_strand = 1 if gene_row['Direction'] == '+' else -1

            if (0 <= gene_start < window_size) or (0 <= gene_end < window_size) or (gene_start <= 0 and gene_end >= window_size):

                gene_start = max(0, gene_start)
                gene_end = min(window_size - 1, gene_end)

                # Gene body is 1 or -1 depending on directionality (0 if no operon present)
                gene_directionality_vector[gene_start:gene_end] = gene_strand

        # Populate promoter vector
        for _, promoter in promoter_df.iterrows():
            promoter_pos = int(promoter['Absolute_Plus_1_Position']) - window_start 
           
            # start and end sites of operons are marked
            if (0 <= promoter_pos < window_size):
                promoter_vector[promoter_pos] = 1

        # Populate terminator vector
        for _, terminator in terminator_df.iterrows():
            terminator_start = int(terminator['Left_End_Position']) - window_start 
            terminator_end = int(terminator['Right_End_Position']) - window_start 
            
            if (0 <= terminator_start < window_size) or (0 <= terminator_end < window_size):

                terminator_start = max(0, terminator_start)
                terminator_end = min(window_size - 1, terminator_end)

                # Termminator is marked
                terminator_vector[terminator_start:terminator_end] = 1

        # Concatenate all vectors
        full_vector = numpy.concatenate((encoded_seq, gene_vector[:, None], promoter_vector[:, None], terminator_vector[:, None], gene_directionality_vector[:, None]), axis=1)
        sequences.append(full_vector)

    return numpy.array(sequences)