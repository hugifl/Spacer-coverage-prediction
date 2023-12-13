import numpy 
import pandas as pd
import HTSeq
from matplotlib import pyplot
from scipy.ndimage import gaussian_filter1d


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

def filter_bamlist(bamlist, counts_df, mincount):
    filtered_bamlist = []
    for bam_file in counts_df.columns[6:]:  
        total_count = counts_df[bam_file].sum()
        if total_count >= mincount:
            filtered_bamlist.append(bam_file)
    print("start of bamlist: " + str(bamlist[:5]))
    print("start of filtered bamlist: " + str(filtered_bamlist[:5]))
    return [bam for bam in bamlist if bam in filtered_bamlist]

def total_count_per_bam(counts_df):
    bam_counts = {}

    for bam_file in counts_df.columns[6:]: 
        total_count = counts_df[bam_file].sum()
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



def process_batch(bam_batch, windows, count_dict, binsize, bam_directory):
    # Initialize DataFrame to store batch coverage data
    batch_coverage_list = []

    for bam_file in bam_batch:
        bam_file_full = bam_directory + bam_file
        bamfile = HTSeq.BAM_Reader(bam_file_full)
        coverage = HTSeq.GenomicArray("auto", stranded=False, typecode="i")
    
        # Read through the bam file and add coverage
        for almnt in bamfile:
            if almnt.aligned:
                coverage[almnt.iv] += 1

        # Calculate coverage for each window and store it
        for window_start, window_end in windows:
            window_iv = HTSeq.GenomicInterval("NC_000913.3", window_start, window_end, ".") # U00096.3
            coverage_array = numpy.fromiter(coverage[window_iv], dtype='i', count=window_end - window_start)
            #coverage_array = coverage_array / count_dict[bam_file]  # Normalize by total number of spacers per bam file          CHANGE MAYBE
            coverage_array, num_bins = bin_coverage(coverage_array, binsize)  # Bin the coverage array
            coverage_row = [window_start, window_end] + coverage_array.tolist()
            batch_coverage_list.append(coverage_row)
            

    # Convert to DataFrame
    batch_coverage_columns = ['Window_Start', 'Window_End'] + [f'Pos_{i}' for i in range(num_bins)]
    batch_coverage_df = pd.DataFrame(batch_coverage_list, columns=batch_coverage_columns)
    return batch_coverage_df

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

def get_gene_spacer_counts(count_df, gene_df): # no used (?)
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

# The normalization factors (gene expression) are calculated from the count matrix with the forward reads also while the coverage is calculated from only reverse reads.
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







        

   