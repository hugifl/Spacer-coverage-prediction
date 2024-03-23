import numpy as np
from utils_sequence import one_hot_encode

# ------------------------------------------------ utility functions ------------------------------------------------
    
def generate_random_dna_sequences_old(num_sequences, seq_length, gc_content):
    assert 0 <= gc_content <= 1, "GC content must be between 0 and 1."
    
    gc_count = int(seq_length * gc_content)
    at_count = seq_length - gc_count  
    
    gc_bases = ['G', 'C'] * gc_count
    at_bases = ['A', 'T'] * at_count
    base_pool = gc_bases + at_bases
    
    np.random.shuffle(base_pool) 
    seq = ''.join(np.random.choice(base_pool, seq_length, replace=True))
    one_hot = one_hot_encode(seq)
    return one_hot

def generate_random_dna_sequences(num_sequences, seq_length, gc_content):
    assert 0 <= gc_content <= 1, "GC content must be between 0 and 1."
    
    # Initialize the array to hold all sequences
    all_sequences_one_hot = np.zeros((num_sequences, seq_length, 4), dtype=int)
    
    for seq_index in range(num_sequences):
        gc_count = int(seq_length * gc_content / 2)  # Divided by 2 to account for both G and C
        at_count = seq_length - 2 * gc_count  # Adjusted for both A and T
        
        # Create a sequence respecting the desired GC content
        bases = ['G', 'C'] * gc_count + ['A', 'T'] * (at_count // 2)
        np.random.shuffle(bases)  # Shuffle to randomize the sequence
        seq = ''.join(bases)
        
        one_hot = one_hot_encode(seq)
        
        # Assign the one-hot encoded sequence to the array
        all_sequences_one_hot[seq_index, :, :] = one_hot  # No need to transpose
        
    return all_sequences_one_hot

def generate_annotation_features(num_sequences, seq_length, len_gene, nu_prom, gene_direction, nu_terminator, terminator_length, gene_position = 'center'):
    X_test_anno = np.zeros((num_sequences, seq_length, 4), dtype=int)
    
    if gene_position == 'center':
        start_gene = (seq_length - len_gene) // 2
        end_gene = start_gene + len_gene

    if gene_position == 'start':
        start_gene = seq_length // 10
        end_gene = start_gene + len_gene

    if gene_position == 'end':
        end_gene = seq_length - (seq_length // 10)
        start_gene = end_gene - len_gene
    
    # Annotate features
    for i in range(num_sequences):
        # Mark gene start and end sites
        X_test_anno[i, start_gene, 0] = 1 
        X_test_anno[i, end_gene - 1, 0] = 1 
        
        # Mark promoter sites
        if gene_direction == '+':
            promoter_pos = max(0, start_gene - nu_prom)
            X_test_anno[i, promoter_pos, 1] = 1
        else:
            promoter_pos = min(seq_length, end_gene + nu_prom)
            X_test_anno[i, promoter_pos, 1] = 1
        
        # Mark the gene body
        if gene_direction == '+':
            X_test_anno[i, start_gene:end_gene, 3] = 1
        else:
            X_test_anno[i, start_gene:end_gene, 3] = -1

        if terminator_length != 0:
            # Mark terminator sites
            if gene_direction == '+':
                terminator_pos = min(seq_length, end_gene + nu_terminator)
                X_test_anno[i, terminator_pos:terminator_pos+terminator_length, 2] = 1
            else:
                terminator_pos = max(0, start_gene - nu_terminator)
                X_test_anno[i, terminator_pos-terminator_length:terminator_pos, 2] = 1

    return X_test_anno

def generate_annotation_features_2_genes(num_sequences, seq_length, len_gene, nu_prom, gene_direction, nu_terminator, terminator_length):
    X_test_anno = np.zeros((num_sequences, seq_length, 4), dtype=int)
    
    start_gene = seq_length // 12
    end_gene = start_gene + len_gene

    
    start_gene_2 = end_gene + 50
    end_gene_2 = start_gene_2 + len_gene

    
    
    # Annotate features
    for i in range(num_sequences):
        # Mark gene start and end sites
        X_test_anno[i, start_gene, 0] = 1 
        X_test_anno[i, end_gene - 1, 0] = 1 
        X_test_anno[i, start_gene_2, 0] = 1 
        X_test_anno[i, end_gene_2 - 1, 0] = 1 
        
        # Mark promoter sites
        if gene_direction == '+':
            promoter_pos = max(0, start_gene - nu_prom)
            X_test_anno[i, promoter_pos, 1] = 1
        else:
            promoter_pos = min(seq_length, end_gene + nu_prom)
            X_test_anno[i, promoter_pos, 1] = 1
        
        # Mark the gene body
        if gene_direction == '+':
            X_test_anno[i, start_gene:end_gene-1, 3] = 1
            X_test_anno[i, start_gene_2:end_gene_2-1, 3] = 1
        else:
            X_test_anno[i, start_gene:end_gene-1, 3] = -1
            X_test_anno[i, start_gene_2:end_gene_2-1, 3] = -1

        if terminator_length != 0:
            # Mark terminator sites
            if gene_direction == '+':
                terminator_pos = min(seq_length, end_gene + nu_terminator)
                X_test_anno[i, terminator_pos:terminator_pos+terminator_length, 2] = 1
            else:
                terminator_pos = max(0, start_gene - nu_terminator)
                X_test_anno[i, terminator_pos-terminator_length:terminator_pos, 2] = 1

    return X_test_anno


def filter_annotation_features_simulated(X_test_anno, annotation_features_to_use):

    feature_index_map = {
    'gene_vector': 0,
    'promoter_vector': 1,
    'terminator_vector': 2,
    'gene_directionality_vector': 3,
    'TU_forward_start_end': 4,
    'TU_reverse_start_end': 5,
    'TU_forward_body': 6,
    'TU_reverse_body': 7,
    'TU_forward_body_cummul': 8,
    'TU_reverse_body_cummul': 9
    }

    indices_to_keep = [feature_index_map[feature] for feature in annotation_features_to_use]

    X_test_anno_filtered = X_test_anno[..., indices_to_keep]

    return X_test_anno_filtered


def generate_dummy_sequences_old(num_sequences, seq_length, dummy_sequence, dummy_sequence_2 = None):
    
    # Initialize the array to hold all sequences
    all_sequences_one_hot = np.zeros((num_sequences, seq_length, 4), dtype=int)
    # calculate number of repetitions of dummy_sequence
    num_reps = seq_length // len(dummy_sequence)
    for seq_index in range(num_sequences):
        bases = list(dummy_sequence) * num_reps
        seq = ''.join(bases)
        
        one_hot = one_hot_encode(seq)
        
        # Assign the one-hot encoded sequence to the array
        all_sequences_one_hot[seq_index, :, :] = one_hot  #
        
    return all_sequences_one_hot
def generate_dummy_sequences(num_sequences, seq_length, dummy_sequence, dummy_sequence_2=None):
    # Initialize the array to hold all sequences
    all_sequences_one_hot = np.zeros((num_sequences, seq_length, 4), dtype=int)
    # Generate the base sequence by repeating the dummy_sequence
    num_reps = seq_length // len(dummy_sequence)
    base_seq = (list(dummy_sequence) * num_reps)[:seq_length]  # Ensure it's cut to seq_length

    for seq_index in range(num_sequences):
        seq = base_seq.copy()  # Start with a fresh copy of the base sequence for each new sequence

        # If dummy_sequence_2 is provided, replace specific stretches
        if dummy_sequence_2 is not None:
            # Calculate the stretch lengths and positions
            stretch_len = 100
            positions = [seq_length // 4 - stretch_len // 2, seq_length // 2 - stretch_len // 2, 4 * seq_length // 5 - stretch_len // 2]
            
            # Generate the stretch from dummy_sequence_2
            stretch_seq = (list(dummy_sequence_2) * (stretch_len // len(dummy_sequence_2) + 1))[:stretch_len]
            
            # Replace the stretches in the sequence
            for pos in positions:
                seq[pos:pos+stretch_len] = stretch_seq

        # Convert the sequence to one-hot encoding
        one_hot = one_hot_encode(''.join(seq))
        
        # Assign the one-hot encoded sequence to the array
        all_sequences_one_hot[seq_index, :, :] = one_hot
        
    return all_sequences_one_hot