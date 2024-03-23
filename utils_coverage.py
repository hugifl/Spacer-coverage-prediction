import numpy 
import pandas as pd
import HTSeq
from matplotlib import pyplot
from scipy.ndimage import gaussian_filter1d
from utils_train_test_data import replace_ones

def get_windows(genome_length,window_size,overlap_size):
    windows = []
    for start in range(1, genome_length, window_size - overlap_size):
        end = start + window_size
        if end > genome_length:
            start = genome_length - window_size
            end = genome_length
        windows.append((start, end))

        if start == genome_length - window_size:
            break
    return windows

def filter_bamlist_old(bamlist, counts_df, mincount, bamfile_start):
    counts_df.columns = [bamfile_start + col.split(bamfile_start, 1)[-1] if bamfile_start in col else col for col in counts_df.columns]
    filtered_bamlist = []
    for bam_file in counts_df.columns[6:]:  
        total_count = counts_df[bam_file].sum()
        if total_count >= mincount:
            filtered_bamlist.append(bam_file)
    print("start of bamlist: " + str(bamlist[:5]))
    print("start of filtered bamlist: " + str(filtered_bamlist[:5]))
    return [bam for bam in bamlist if bam in filtered_bamlist]

def filter_bamlist(bamlist, counts_df, mincount, bamfile_start):
    # Update column names in counts_df
    counts_df.columns = [
        bamfile_start + col.split(bamfile_start, 1)[-1] if bamfile_start in col else col
        for col in counts_df.columns
    ]

    # Initialize an empty list for filtered BAM files
    filtered_bamlist = []

    # Iterate over the columns of counts_df starting from the 7th column
    for bam_file in counts_df.columns[6:]:
        total_count = counts_df[bam_file].sum()
        if total_count >= mincount:
            filtered_bamlist.append(bam_file)
        else:
            # Delete the column if total_count is less than mincount
            counts_df.drop(bam_file, axis=1, inplace=True)

    # Print the first few elements of bamlist and filtered_bamlist for debugging
    print("Start of bamlist: " + str(bamlist[:5]))
    print("Start of filtered bamlist: " + str(filtered_bamlist[:5]))

    # Filter the bamlist to include only those BAM files that are in filtered_bamlist
    filtered_bamlist_final = [bam for bam in bamlist if bam in filtered_bamlist]
    counts_df.reset_index(drop=True, inplace=True)

    # Return the filtered list and the updated counts_df
    return filtered_bamlist_final, counts_df

def filter_bamlist_2(count_dict, threshold):
    # Convert dict_keys object to a list for modification
    filtered_bamlist = list(count_dict.keys())

    for key, value in count_dict.items():
        if value < threshold:
            filtered_bamlist.remove(key)

    return filtered_bamlist

def total_count_per_bam(counts_df, bamfile_start):
    bam_counts = {}

    for bam_file in counts_df.columns[6:]: 
        total_count = counts_df[bam_file].sum()
        bam_file = bamfile_start + bam_file.split(bamfile_start, 1)[-1] if bamfile_start in bam_file else bam_file
        bam_counts[bam_file] = total_count

    return bam_counts


#def lowest_expressed_genes_per_bamlist(bamlist, counts_df, gene_perc):   # not used
#    gene_dict = {}
#    for _, row in counts_df.iterrows():
#        spacer_count = 0
#        gene_name = row['Geneid']  
#        for bam_file in counts_df.columns[6:]:
#            if bam_file in bamlist:
#                spacer_count += row[bam_file]
#        gene_dict[gene_name] = spacer_count
#    
#    sorted_genes = sorted(gene_dict, key=lambda k: gene_dict[k])
#    number_to_be_dropped = int(len(sorted_genes) * (gene_perc / 100))
#    return sorted_genes[:number_to_be_dropped]
def remove_lowest_expressed_TUs(count_df, lowest_expressed_TUs):
    adjusted_count_df = count_df[~count_df['TU_Name'].isin(lowest_expressed_TUs)]
    adjusted_count_df.reset_index(drop=True, inplace=True)
    return adjusted_count_df

def filter_windows(windows_coverage_df, filtered_genes, gene_df):

    indices_to_drop = set()
    counter = 0

    for gene in filtered_genes:
        counter += 1
        gene_data = gene_df[gene_df['Gene_Name'].str.contains(gene.replace("-", ""), na=False)]

        # Check if gene_data has only one row
        if len(gene_data) == 1:
            gene_left_end = int(gene_data['Left'].iloc[0])
            gene_right_end = int(gene_data['Right'].iloc[0])

            for index, row in windows_coverage_df.iterrows():
                check_left = gene_left_end >= int(row['Window_Start']) and gene_left_end <= int(row['Window_End'])
                check_right = gene_right_end >= int(row['Window_Start']) and gene_right_end <= int(row['Window_End'])

                if check_left or check_right:
                    indices_to_drop.add(index)
    
    windows_coverage_df.drop(indices_to_drop, inplace=True)
    windows_coverage_df.reset_index(drop=True, inplace=True)

    return windows_coverage_df


def bin_coverage(coverage_array, binsize):
    num_bins = int(numpy.ceil(len(coverage_array) / binsize))
    binned_coverage = numpy.zeros(num_bins)

    for i in range(num_bins):
        start_index = i * binsize
        end_index = min((i + 1) * binsize, len(coverage_array))
        binned_coverage[i] = numpy.mean(coverage_array[start_index:end_index])
    return binned_coverage, num_bins



def process_batch(bam_batch, windows, reference_genome, binsize, bam_directory):
    # Initialize DataFrame to store batch coverage data
    batch_coverage_list = []
    aligned_read_count = 0
    window_aligned_read_count = 0
    total_read_count = 0

    for bam_file in bam_batch:
        bam_file_full = bam_directory + bam_file
        bamfile = HTSeq.BAM_Reader(bam_file_full)
        coverage = HTSeq.GenomicArray("auto", stranded=False, typecode="i")

        

        # Read through the bam file and add coverage
        for almnt in bamfile:
            total_read_count += 1
            if almnt.aligned:
                coverage[almnt.iv] += 1
                aligned_read_count += 1

        # Calculate coverage for each window and store it
        for window_start, window_end in windows:
            window_iv = HTSeq.GenomicInterval(reference_genome, window_start, window_end, ".") 
            window_ref = reference_genome
            window_start_pos = window_start
            window_end_pos = window_end
    
            # Iterate over alignments in the window
            for almnt in bamfile.fetch(window_ref, window_start_pos, window_end_pos):
                if almnt.aligned:
                    window_aligned_read_count += 1
            coverage_array = numpy.fromiter(coverage[window_iv], dtype='i', count=window_end - window_start  ) 
            #coverage_array = coverage_array / count_dict[bam_file]  # Normalize by total number of spacers per bam file          CHANGE MAYBE
            coverage_array, num_bins = bin_coverage(coverage_array, binsize)  # Bin the coverage array
            coverage_row = [window_start, window_end] + coverage_array.tolist()
            batch_coverage_list.append(coverage_row)
            

    # Convert to DataFrame
    batch_coverage_columns = ['Window_Start', 'Window_End'] + [f'Pos_{i}' for i in range(num_bins)]
    batch_coverage_df = pd.DataFrame(batch_coverage_list, columns=batch_coverage_columns)
    return batch_coverage_df, aligned_read_count, window_aligned_read_count, total_read_count

def process_batch_TU(bam_batch, count_df, reference_genome, bam_directory, pad_symbol):
    pad_symbol = float(pad_symbol)
    # Determine the length of the longest TU
    max_length = count_df['Length'].max() - 1
    aligned_read_count = 0
    TU_aligned_read_count = 0
    total_read_count = 0

    # Initialize list to store TU coverage data
    tu_coverage_list = []

    for bam_file in bam_batch:
        bam_file_full = bam_directory + bam_file
        bamfile = HTSeq.BAM_Reader(bam_file_full)

        for almnt in bamfile:
            total_read_count += 1
            if almnt.aligned:
                aligned_read_count += 1

        for _, row in count_df.iterrows():
            tu_name = row['Geneid']
            start = int(row['Start'])
            end = int(row['End'])
            direction = row['Strand']

            # Define genomic interval for the TU
            tu_iv = HTSeq.GenomicInterval(reference_genome, start, end, ".")

            # Create a GenomicArray for coverage calculation
            coverage = HTSeq.GenomicArray("auto", stranded=False, typecode="i")
            
            # Fetch alignments and calculate coverage
            for almnt in bamfile.fetch(reference_genome, start, end):
                if almnt.aligned:
                    coverage[almnt.iv] += 1
                    TU_aligned_read_count += 1
            
            # Extract coverage array for the TU
            coverage_array = numpy.fromiter(coverage[tu_iv], dtype='i', count=end - start)
            coverage_array = coverage_array.astype(float) 

            # Reverse the coverage array if the TU is in '-' direction
            if direction == '-':
                coverage_array = coverage_array[::-1]
            
            # Pad the coverage array for shorter TUs
            padded_coverage_array = numpy.pad(coverage_array, (0, max_length - len(coverage_array)), 'constant', constant_values=pad_symbol)

            # Add TU coverage data to the list
            tu_coverage_list.append([tu_name, start, end, direction] + padded_coverage_array.tolist())

    # Define DataFrame columns
    coverage_columns = ['TU_Name', 'Start', 'End', 'Direction'] + [f'Pos_{i}' for i in range(max_length)]
    
    # Convert list to DataFrame
    tu_coverage_df = pd.DataFrame(tu_coverage_list, columns=coverage_columns)
    print("head of tu coverage df: ", tu_coverage_df.head())
    return tu_coverage_df, aligned_read_count, TU_aligned_read_count, total_read_count

def define_peaks(coverage_data, local_threshold=0.3, global_threshold=0.001, window_info= False):
    if not window_info:
        binary_peaks = numpy.zeros_like(coverage_data)
        for i, window in enumerate(coverage_data):
            window_max = numpy.max(window)
            local_peak_threshold = window_max * local_threshold
            effective_threshold = max(local_peak_threshold, global_threshold)
            binary_peaks[i] = numpy.where(window >= effective_threshold, 1, 0)
    else:
        binary_peaks = numpy.zeros_like(coverage_data)
        for i, window in enumerate(coverage_data):
            window_coverage = window[2:]  # Exclude the first two columns which are window info
            window_max = numpy.max(window_coverage)
            local_peak_threshold = window_max * local_threshold
            effective_threshold = max(local_peak_threshold, global_threshold)
            binary_window = numpy.where(window_coverage >= effective_threshold, 1, 0)
            binary_peaks[i, 2:] = binary_window  # Update only the coverage part of the row
            binary_peaks[i, :2] = window[:2]  # Retain the window start and end info

    return binary_peaks

def define_peaks_single_window(coverage_data, local_threshold=0.3, global_threshold=0.001):
    binary_peaks = numpy.zeros_like(coverage_data)
    window_max = numpy.max(coverage_data)
    local_peak_threshold = window_max * local_threshold
    effective_threshold = max(local_peak_threshold, global_threshold)
    binary_peaks = numpy.where(coverage_data >= effective_threshold, 1, 0)
    return binary_peaks

def get_operon_spacer_counts(count_df, gene_df, operon_df):
    # Create a dictionary to store operon spacer counts
    operon_spacer_counts = {}
    no_operons = count_df.shape[0]
    count = 0
    for _, operon_row in operon_df.iterrows():
        count += 1
        print("fraction done: " + str(count/no_operons))
        operon_name = operon_row['operonName']
        operon_start = min(int(operon_row['firstGeneLeftPos']), int(operon_row['lastGeneRightPos']))
        operon_end = max(int(operon_row['firstGeneLeftPos']), int(operon_row['lastGeneRightPos']))

        # Initialize spacer count for this operon
        operon_spacer_count = 0

        # Iterate through each gene in the gene_df
        for _, gene_row in gene_df.iterrows():
            if gene_row['leftEndPos'] == 'None' or gene_row['rightEndPos']  == 'None':
                continue
            gene_name = gene_row['geneName'].split('(')[0].split('-')[0]  # Extracting 'insB1' from 'insB1(b0021)'
            gene_start = min(int(gene_row['leftEndPos']), int(gene_row['rightEndPos']))
            gene_end = max(int(gene_row['leftEndPos']), int(gene_row['rightEndPos']))

            if gene_start >= operon_start and gene_end <= operon_end:
                matched_genes = [row['Geneid'] for _, row in count_df.iterrows() if gene_name in row['Geneid'].replace("-", "")]

                for matched_gene in matched_genes:
                    mask = count_df['Geneid'] == matched_gene
                    selected_rows = count_df[mask]
                    operon_spacer_count += selected_rows.iloc[:, 6:].sum().sum()


                # Store the spacer count for this operon
                operon_spacer_counts[operon_name] = {'start': operon_start, 'end': operon_end, 'spacer_count': operon_spacer_count}

    # Convert the dictionary to a DataFrame
    operon_spacer_counts_df = pd.DataFrame.from_dict(operon_spacer_counts, orient='index')
    operon_spacer_counts_df.reset_index(inplace=True)
    operon_spacer_counts_df.rename(columns={'index': 'Operon_Name'}, inplace=True)

    return operon_spacer_counts_df

def get_gene_spacer_counts(count_df, gene_df): # not used (?)
    # Create a dictionary to store gene spacer counts
    gene_spacer_counts_normalized = {}
    no_genes = count_df.shape[0]
    count = 0

    for _, gene_row in gene_df.iterrows():
        count += 1
        gene_spacer_count = 0
        
        if gene_row['leftEndPos'] == 'None' or gene_row['rightEndPos']  == 'None':
            continue
        gene_name = gene_row['geneName'].split('(')[0].split('-')[0]  # Extracting 'insB1' from 'insB1(b0021)'
        gene_start = min(int(gene_row['leftEndPos']), int(gene_row['rightEndPos']))
        gene_end = max(int(gene_row['leftEndPos']), int(gene_row['rightEndPos']))
        gene_length = gene_end - gene_start + 1
        matched_genes = [row['Geneid'] for _, row in count_df.iterrows() if gene_name in row['Geneid'].replace("-", "")]
        for matched_gene in matched_genes:
            mask = count_df['Geneid'] == matched_gene
            selected_rows = count_df[mask]
            gene_spacer_count += selected_rows.iloc[:, 6:].sum().sum()
            normalized_gene_spacer_count = gene_spacer_count / gene_length
        
        # Store the spacer count for this operon
        gene_spacer_counts_normalized[gene_name] = {'start': gene_start, 'end': gene_end, 'spacer_count': normalized_gene_spacer_count}

    # Convert the dictionary to a DataFrame
    gene_spacer_counts_normalized_df = pd.DataFrame.from_dict(gene_spacer_counts_normalized, orient='index')
    gene_spacer_counts_normalized_df.reset_index(inplace=True)
    gene_spacer_counts_normalized_df.rename(columns={'index': 'Gene_Name'}, inplace=True)

    return gene_spacer_counts_normalized_df

# The normalization factors (gene expression) are calculated from the count matrix.
def get_normalized_spacer_counts_per_gene(counts_df, gene_perc, count_dict):
    gene_dict = {}
    gene_dict_df = {}
    no_bam_files = len(count_dict)
    for _, row in counts_df.iterrows():
        spacer_count = 0
        gene_name = row['Geneid']  
        gene_start = int(row['Start'])
        gene_end = int(row['End'])
        gene_start, gene_end = min(gene_start, gene_end), max(gene_start, gene_end)                                                                           
        gene_length = float(gene_end - gene_start) + 1
        for bam_file in counts_df.columns[6:]:
            spacer_count += (row[bam_file]) / (float(count_dict[bam_file])/1000000) #              changed from per 10000 reads to per 1000000 read 
        gene_dict[gene_name] = ((spacer_count/(gene_length/1000))/no_bam_files)                   # spacer count in spacers per kilobase (gene length) per 10000 mapped spacers averaged over all samples
        gene_dict_df[gene_name] = {'start': gene_start, 'end': gene_end, 'spacer_count': ((spacer_count/(gene_length/1000))/no_bam_files)}
    sorted_genes = sorted(gene_dict, key=lambda k: gene_dict[k])
    number_to_be_dropped = int(len(sorted_genes) * (gene_perc / 100))
    normalized_gene_spacer_counts_df = pd.DataFrame.from_dict(gene_dict_df, orient='index')
    normalized_gene_spacer_counts_df.reset_index(inplace=True)
    normalized_gene_spacer_counts_df.rename(columns={'index': 'Gene_Name'}, inplace=True)
    return sorted_genes[:number_to_be_dropped], normalized_gene_spacer_counts_df

def get_normalized_spacer_counts_per_TU(counts_df, TU_perc, count_dict):
    TU_dict = {}
    TU_dict_df = {}
    no_bam_files = len(count_dict)
    for _, row in counts_df.iterrows():
        spacer_count = 0
        TU_name = row['Geneid']  
        TU_start = int(row['Start'])
        TU_end = int(row['End'])
        TU_start, TU_end = min(TU_start, TU_end), max(TU_start, TU_end)                                                                           
        TU_length = float(TU_end - TU_start) + 1
        for bam_file in counts_df.columns[6:]:
            spacer_count += (row[bam_file]) / (float(count_dict[bam_file])/1000000) #              changed from per 10000 reads to per 1000000 read 
        TU_dict[TU_name] = ((spacer_count/(TU_length/1000))/no_bam_files)                   # spacer count in spacers per kilobase (gene length) per 10000 mapped spacers averaged over all samples
        TU_dict_df[TU_name] = {'start': TU_start, 'end': TU_end, 'spacer_count': ((spacer_count/(TU_length/1000))/no_bam_files)}
    sorted_TUs = sorted(TU_dict, key=lambda k: TU_dict[k])
    number_to_be_dropped = int(len(sorted_TUs) * (TU_perc / 100))
    normalized_TU_spacer_counts_df = pd.DataFrame.from_dict(TU_dict_df, orient='index')
    normalized_TU_spacer_counts_df.reset_index(inplace=True)
    normalized_TU_spacer_counts_df.rename(columns={'index': 'TU_Name'}, inplace=True)
    return sorted_TUs[:number_to_be_dropped], normalized_TU_spacer_counts_df


def normalize_coverage_per_gene(coverage_data, gene_spacer_counts_df, no_bin, binsize): 
    normalized_coverage_data = coverage_data.copy()
    #windows_to_delete = [] # If there is some window that isn't covered by any gene, it is deleted.
    counter = 0
    nrow = coverage_data.shape[0]
    for index, row in enumerate(coverage_data):
        counter += 1
        #print("perc. done: " + str(100*(counter/nrow)))
        window_start = int(row[0]) 
        window_end = int(row[1]) 
        #print("WINDOW: " + str(window_start)+" to " + str(window_end))
        normalization_factors = numpy.ones(int(no_bin))

        for _, gene in gene_spacer_counts_df.iterrows():
            gene_name = gene['Gene_Name']
            gene_start = min(int(gene['start']), int(gene['end']))
            gene_end = max(int(gene['start']), int(gene['end']))

            # Checking cases where an gene spans the whole window
            if (gene_start <= window_start) and (gene_end >= window_end): 
                #print("spanned the whole winodw: ")
                #print(gene_name)
                #print("gene start: " + str(gene_start))
                #print("gene end: " + str(gene_end))
                normalization_factors[:] = float(gene['spacer_count'])
                break
            
            # Checking cases where an gene starts before the window and ends within the window
            if (gene_start <= window_start) and (window_start  <= gene_end <= window_end):
                #print("ended in window:")
                #print(gene_name)
                #print("gene start: " + str(gene_start))
                #print("gene end: " + str(gene_end))
                end_index = int((gene_end - window_start)/binsize)
                normalization_factors[:end_index] = float(gene['spacer_count'])
            
            # Checking cases where an gene starts and ends within the window
            if (window_start  <= gene_start <= window_end) and (window_start  <= gene_end <= window_end):
                #print("started and ended in window:")
                #print(gene_name)
                #print("gene start: " + str(gene_start))
                #print("gene end: " + str(gene_end))
                start_index = int((gene_start - window_start)/binsize)
                end_index = int((gene_end - window_start)/binsize)
                normalization_factors[start_index:end_index] = float(gene['spacer_count'])
            
            # Checking cases where an gene starts within the window and ends after the window
            if (window_start  <= gene_start <= window_end) and (gene_end >= window_end):
                #print("started in window:")
                #print(gene_name)
                #print("gene start: " + str(gene_start))
                #print("gene end: " + str(gene_end))
                start_index = int((gene_start - window_start)/binsize)
                normalization_factors[start_index:] = float(gene['spacer_count'])
            
        # Normalizing regions between gene with the count of the closest gene
        #print("normalizataion factors before filling ones:")
        #print(normalization_factors)
        normalization_factors, status = replace_ones(normalization_factors)
        #print("normalizataion factors after filling ones:")
        #print(normalization_factors)
        #if status == "only_ones":
        #    #print("DANGER DANGER DANGER ONLY ONEES ALAALALALALLAA")
        #    windows_to_delete.append(index)
        
        normalized_coverage_data[index,2:] = (coverage_data[index,2:]) / normalization_factors      # 10000 scaling factor removed
        #print("coverage:")
        #print(coverage_data[index,:])
        #print("normalized coverage:")
        #print(normalized_coverage_data[index,:])

    #normalized_coverage_data = numpy.delete(normalized_coverage_data, windows_to_delete, axis=0)
    #sequence_data = numpy.delete(sequence_data, windows_to_delete, axis=0)

    return normalized_coverage_data

def normalize_TU_coverage_per_TU(coverage_data, TU_spacer_counts_df, pad_symbol):
    # Initialize a normalized coverage DataFrame
    normalized_coverage_data = coverage_data.copy()
    
    # Preprocess to create a lookup for spacer counts by TU_Name
    spacer_count_lookup = TU_spacer_counts_df.set_index('TU_Name')['spacer_count'].to_dict()
    
    # Iterate through each row in the coverage DataFrame
    for index, row in normalized_coverage_data.iterrows():
        # Initialize normalization_factors with the spacer_count of the current TU
        tu_length = row['End'] - row['Start']
        current_spacer_count = spacer_count_lookup[row['TU_Name']]
        normalization_factors = numpy.full(tu_length, current_spacer_count)
        
        # Adjust normalization_factors for overlaps
        for _, other_row in TU_spacer_counts_df.iterrows():
            # Skip the current TU
            if other_row['TU_Name'] == row['TU_Name']:
                continue
            
            # Check for overlap and adjust normalization_factors accordingly
            overlap_start = max(row['Start'], other_row['start'])
            overlap_end = min(row['End'], other_row['end'])
            
            if overlap_start < overlap_end:  # There's an overlap
                other_spacer_count = spacer_count_lookup[other_row['TU_Name']]
               
                # Update normalization_factors for the overlap region
                normalization_factors[overlap_start - row['Start']:overlap_end - row['Start']] += other_spacer_count

        if row['Direction'] == '-':
                    normalization_factors = normalization_factors[::-1]
        
        # Normalize coverage data using normalization_factors
        for pos_index, col_name in enumerate(row.index[row.index.str.startswith('Pos_')]):
            if row[col_name] == pad_symbol:
                break  # Stop normalizing when pad_symbol is encountered
            
            if pos_index < len(normalization_factors):  
                normalized_value = row[col_name] / normalization_factors[pos_index]
                normalized_coverage_data.at[index, col_name] = normalized_value
    
    return normalized_coverage_data

def normalize_coverage_per_TU(coverage_data, TU_spacer_counts_df, no_bin, binsize):        
    normalized_coverage_data = coverage_data.copy()
    #windows_to_delete = [] # If there is some window that isn't covered by any gene, it is deleted.
    counter = 0
    nrow = coverage_data.shape[0]
    for index, row in enumerate(coverage_data):
        counter += 1
        #print("perc. done: " + str(100*(counter/nrow)))
        window_start = int(row[0]) 
        window_end = int(row[1]) 
        #print("WINDOW: " + str(window_start)+" to " + str(window_end))
        normalization_factors = numpy.ones(int(no_bin))

        for _, TU in TU_spacer_counts_df.iterrows():
            TU_name = TU['TU_Name']
            TU_start = min(int(TU['start']), int(TU['end']))
            TU_end = max(int(TU['start']), int(TU['end']))

            # Checking cases where an gene spans the whole window
            if (TU_start <= window_start) and (TU_end >= window_end): 
                normalization_factors[:] += float(TU['spacer_count'])
            
            # Checking cases where an gene starts before the window and ends within the window
            if (TU_start <= window_start) and (window_start  <= TU_end <= window_end):
                end_index = int((TU_end - window_start)/binsize)
                normalization_factors[:end_index] += float(TU['spacer_count'])
            
            # Checking cases where an gene starts and ends within the window
            if (window_start  <= TU_start <= window_end) and (window_start  <= TU_end <= window_end):
                start_index = int((TU_start - window_start)/binsize)
                end_index = int((TU_end - window_start)/binsize)
                normalization_factors[start_index:end_index] += float(TU['spacer_count'])
            
            # Checking cases where an gene starts within the window and ends after the window
            if (window_start  <= TU_start <= window_end) and (TU_end >= window_end):
                start_index = int((TU_start - window_start)/binsize)
                normalization_factors[start_index:] += float(TU['spacer_count'])
            
        # Normalizing regions between gene with the count of the closest TU
        normalization_factors, status = replace_ones(normalization_factors)
        
        normalized_coverage_data[index,2:] = (coverage_data[index,2:]) / normalization_factors      # 10000 scaling factor removed
    return normalized_coverage_data


def normalize_coverage_for_tot_aligned_reads_old(coverage_df, library_size, expected_aligned_reads_per_library):
    # Create a copy of the DataFrame to avoid modifying the original data
    coverage_df_normalized = coverage_df.copy()
    coverage_df_normalized.iloc[:, 2:] = coverage_df.iloc[:, 2:] * (expected_aligned_reads_per_library / library_size)
    
    return coverage_df_normalized

def normalize_coverage_for_tot_aligned_reads_TU(coverage_data, library_size, expected_aligned_reads_per_library):
    normalization_factor = float(expected_aligned_reads_per_library / library_size)
    coverage_normalized = coverage_data.copy()
    ncols = coverage_normalized.shape[1]
    print("ncols: " + str(ncols))
    # Iterate through each row to apply normalization
    for index, row in coverage_normalized.iterrows():
        # Calculate the length from 'Start' and 'End'
        length = row['End'] - row['Start']
        #print("length: " + str(length))
        
        # Columns to be normalized start at index 5 and end at index 5 + length
        start_col_index = 4
        end_col_index = start_col_index + length - 1
        # Apply normalization for the specified range
        # Ensure the range does not exceed the DataFrame's column limits
        #print("end_col_index before: " + str(end_col_index))
        end_col_index = min(end_col_index, coverage_normalized.shape[1])
        # Normalize the coverage data for the relevant columns
        #print("start_col_index: " + str(start_col_index))
        #print("end_col_index after: " + str(end_col_index))
        #print("TU length: " + str(length))
        #print("TU name: " + str(row['TU_Name']))
        #print("length of row: " + str(len(row)))
        #print("start of row: " + str(row[:40]))
        # Iterate over the specified slice and check data types
        for col in range(start_col_index, end_col_index):
            value = coverage_normalized.iloc[index, col]
            if not isinstance(value, (int, float, numpy.number)):
                print(f"Non-numeric value found: {value} at Row {index}, Column {col}")
        coverage_normalized.iloc[index, start_col_index:end_col_index] = coverage_normalized.iloc[index, start_col_index:end_col_index] * normalization_factor
    return coverage_normalized



def normalize_coverage_for_tot_aligned_reads(coverage_data, library_size, expected_aligned_reads_per_library):
    # Check if the input is a pandas DataFrame
    if isinstance(coverage_data, pd.DataFrame):
        # For DataFrame, create a copy and use iloc for indexing
        coverage_normalized = coverage_data.copy()
        coverage_normalized.iloc[:, 2:] *= (expected_aligned_reads_per_library / library_size)
    elif isinstance(coverage_data, numpy.ndarray):
        # For NumPy array, use slicing for indexing
        coverage_normalized = coverage_data.copy()
        coverage_normalized[:, 2:] *= (expected_aligned_reads_per_library / library_size)
    else:
        raise TypeError("Input must be a pandas DataFrame or a NumPy array")

    return coverage_normalized


def expand_count_df(counts_df, OUTDIR = False):
    expanded_rows = []

    for _, row in counts_df.iterrows():
        gene_name = row['Geneid']
        gene_start = row['Start']
        gene_end = row['End']
        chromosome = row['Chr']
        direction = row['Strand']

        try:
            # Test if gene_start and gene_end can be converted to float (i.e., they don't contain ';')
            float(gene_start)
            float(gene_end)
            # If conversion is successful, append the row as is
            expanded_rows.append(row)
        except ValueError:
            # Split gene_start and gene_end values by ';' if they contain multiple values
            gene_starts = str(gene_start).split(';')
            gene_ends = str(gene_end).split(';')
            chromosome = str(chromosome).split(';')
            direction = str(direction).split(';')

            # Ensure the lengths of starts and ends are equal
            if len(gene_starts) != len(gene_ends):
                print(f"Unequal number of starts and ends for gene {gene_name}. Skipping this row.")
                continue

            # Process the columns from the 6th column onwards
            gene_counts = row[6:]

            # Create new rows for each start and end pair
            for i, (start, end) in enumerate(zip(gene_starts, gene_ends)):
                new_row = row.copy()
                new_row['Geneid'] = f'{gene_name}_{i+1}' if i > 0 else gene_name
                new_row['Start'] = int(start)
                new_row['End'] = int(end)
                new_row['Chr'] = chromosome[i]
                new_row['Strand'] = direction[i]
                new_row['Length'] = int(end) - int(start) + 1

                # Distribute counts equally among the rows
                for count_col in gene_counts.index:
                    new_row[count_col] = float(gene_counts[count_col]) / len(gene_starts)

                expanded_rows.append(new_row)

    # Create a new DataFrame from the expanded rows
    expanded_df = pd.DataFrame(expanded_rows)
    if OUTDIR:
        # Save the expanded DataFrame
        expanded_df.to_csv(f"{OUTDIR}/expanded_counts.csv", index=False)

    return expanded_df



def replace_summed_pads(df, pad_symbol):
    df_padded = df.copy()
    total_coverage_cols = df_padded.shape[1] - 2  
    for index, row in df_padded.iterrows():
        tu_length = row['End'] - row['Start']
   
        pad_start_index = 2 + tu_length  # Adjusting for 'Start' and 'End' columns
        
        if pad_start_index < total_coverage_cols:
            pad_start_index = int(numpy.clip(pad_start_index, 0, total_coverage_cols))
            
            # Replace values beyond the TU length with pad_symbol
            df_padded.iloc[index, pad_start_index:] = pad_symbol
            
    return df_padded