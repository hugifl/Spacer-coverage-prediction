import numpy 
import pandas as pd
import HTSeq
from matplotlib import pyplot
from sklearn.utils import shuffle

def get_windows(genome_length,window_size,overlap_size):
    windows = []
    for start in range(0, genome_length, window_size - overlap_size):
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

    return [bam for bam in bamlist if bam in filtered_bamlist]

def total_count_per_bam(counts_df):
    bam_counts = {}

    for bam_file in counts_df.columns[6:]: 
        total_count = counts_df[bam_file].sum()
        bam_counts[bam_file] = total_count

    return bam_counts

def lowest_expressed_genes(counts_df, gene_perc):
    gene_dict = {}
    for _, row in counts_df.iterrows():
        spacer_count = 0
        gene_name = row['Geneid']  
        gene_start = int(row['Start'])
        gene_end = int(row['End'])
        gene_start, gene_end = min(gene_start, gene_end), max(gene_start, gene_end)                                                                           
        gene_length = gene_end - gene_start + 1
        for bam_file in counts_df.columns[6:]:
            spacer_count += row[bam_file]
        gene_dict[gene_name] = spacer_count/gene_length
    
    sorted_genes = sorted(gene_dict, key=lambda k: gene_dict[k])
    number_to_be_dropped = int(len(sorted_genes) * (gene_perc / 100))
    return sorted_genes[:number_to_be_dropped]

def lowest_expressed_genes_per_bamlist(bamlist, counts_df, gene_perc):   # not used
    gene_dict = {}
    for _, row in counts_df.iterrows():
        spacer_count = 0
        gene_name = row['Geneid']  
        for bam_file in counts_df.columns[6:]:
            if bam_file in bamlist:
                spacer_count += row[bam_file]
        gene_dict[gene_name] = spacer_count
    
    sorted_genes = sorted(gene_dict, key=lambda k: gene_dict[k])
    number_to_be_dropped = int(len(sorted_genes) * (gene_perc / 100))
    return sorted_genes[:number_to_be_dropped]

def filter_windows(windows_coverage_df, filtered_genes, gene_df):
    print("Filtering start")
    print(len(filtered_genes))
    indices_to_drop = set()
    counter = 0

    for gene in filtered_genes:
        counter += 1
        print('Gene:', gene)
        print('Fraction done:', counter / len(filtered_genes))

        gene_data = gene_df[gene_df['geneName'].str.contains(gene.replace("-", ""), na=False)]

        # Check if gene_data has only one row
        if len(gene_data) == 1:
            gene_left_end = int(gene_data['leftEndPos'].iloc[0])
            gene_right_end = int(gene_data['rightEndPos'].iloc[0])

            for index, row in windows_coverage_df.iterrows():
                check_left = gene_left_end >= int(row['Window_Start']) and gene_left_end <= int(row['Window_End'])
                check_right = gene_right_end >= int(row['Window_Start']) and gene_right_end <= int(row['Window_End'])

                if check_left or check_right:
                    indices_to_drop.add(index)
        else:
            print(f"Multiple or no entries found for gene {gene}. Entries count: {len(gene_data)}")
    
    print('indices to drop: ' + str(len(indices_to_drop)))
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


def dataframe_to_2darray(df):
    coverage_data = df.drop(columns=['Window_Start', 'Window_End'])

    coverage_array = coverage_data.to_numpy()

    return coverage_array

def dataframe_to_2darray_keep_window_information(df):
    coverage_array = df.to_numpy()

    return coverage_array

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
            window_iv = HTSeq.GenomicInterval("U00096.3", window_start, window_end, ".")
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

def get_gene_spacer_counts(count_df, gene_df):
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

# A function replacing 1s in a vector with the closest non-1 value
def replace_ones(vector):

    if all(element == 1 for element in vector):
        return vector, 'only_ones'
    
    finished = False
    counter = 0
    while finished == False:
        change_made_right = False
        one_counter = 0
        if counter >50:
            finished = True
        for i in range(len(vector)):
            if vector[i] == 1:
                one_counter += 1
                if i == 0:
                    if vector[i+1] != 1:
                        vector[i] = vector[i+1]
                elif i == (len(vector) - 1):
                    if vector[i-1] != 1:
                        vector[i] = vector[i-1]
                else:
                    if vector[i-1] != 1:
                        if change_made_right == False:
                            vector[i] = vector[i-1]
                            change_made_right = True
                        else:
                            change_made_right = False
                    if vector[i+1] != 1:
                        vector[i] = vector[i+1]
        if one_counter == 0:
            finished = True
    return vector, 'replacement finished'



def normalize_coverage_per_operon(coverage_data, sequence_data, operon_spacer_counts_df, no_bin, binsize):
    normalized_coverage_data = coverage_data.copy()
    windows_to_delete = [] # If there is some window that isn't covered by any operon (an annotation mistake), it is deleted.
    counter = 0
    nrow = coverage_data.shape[0]
    for index, row in enumerate(coverage_data):
        counter += 1
        #print("perc. done: " + str(100*(counter/nrow)))

        window_start = int(row[0]) + 1
        window_end = int(row[1]) + 1
        #print("WINDOW: " + str(window_start)+" to " + str(window_end))
        normalization_factors = numpy.ones(no_bin)

        for _, operon in operon_spacer_counts_df.iterrows():
            operon_name = operon['Operon_Name']
            operon_start = min(int(operon['start']), int(operon['end']))
            operon_end = max(int(operon['start']), int(operon['end']))

            # Checking cases where an operon spans the whole window
            if (operon_start <= window_start) and (operon_end >= window_end): 
                #print(operon_name)
                normalization_factors[:] = int(operon['spacer_count'])
                break
            
            # Checking cases where an operon starts before the window and ends within the window
            if (operon_start <= window_start) and (window_start  <= operon_end <= window_end):
                #print("ended in window:")
                #print(operon_name)
                end_index = int((operon_end - window_start)/binsize)
                normalization_factors[:end_index] = int(operon['spacer_count'])
            
            # Checking cases where an operon starts and ends within the window
            if (window_start  <= operon_start <= window_end) and (window_start  <= operon_end <= window_end):
                #print("started and ended in window:")
                #print(operon_name)
                start_index = int((operon_start - window_start)/binsize)
                end_index = int((operon_end - window_start)/binsize)
                normalization_factors[start_index:end_index] = int(operon['spacer_count'])
            
            # Checking cases where an operon starts within the window and ends after the window
            if (window_start  <= operon_start <= window_end) and (operon_end >= window_end):
                #print("started in window:")
                #print(operon_name)
                start_index = int((operon_start - window_start)/binsize)
                normalization_factors[start_index:] = int(operon['spacer_count'])
            
        # Normalizing regions between operons with the count of the closest operon
        #print("normalizataion factors before filling ones:")
        #print(normalization_factors)
        normalization_factors, status = replace_ones(normalization_factors)
        #print("normalizataion factors after filling ones:")
        #print(normalization_factors)
        if status == "only_ones":
            #print("DANGER DANGER DANGER ONLY ONEES ALAALALALALLAA")
            windows_to_delete.append(index)
        
        normalized_coverage_data[index,2:] = (10000000 * coverage_data[index,2:]) / normalization_factors
        #print("coverage:")
        #print(coverage_data[index,:])
        #print("normalized coverage:")
        #print(normalized_coverage_data[index,:])

    normalized_coverage_data = numpy.delete(normalized_coverage_data, windows_to_delete, axis=0)
    normalized_coverage_data_no_window = normalized_coverage_data[:,2:]
    sequence_data = numpy.delete(sequence_data, windows_to_delete, axis=0)

    return normalized_coverage_data, normalized_coverage_data_no_window, sequence_data

def normalize_coverage_per_gene(coverage_data, sequence_data, gene_spacer_counts_df, no_bin, binsize): 
    normalized_coverage_data = coverage_data.copy()
    windows_to_delete = [] # If there is some window that isn't covered by any gene (an annotation mistake), it is deleted.
    counter = 0
    nrow = coverage_data.shape[0]
    for index, row in enumerate(coverage_data):
        counter += 1
        #print("perc. done: " + str(100*(counter/nrow)))
        window_start = int(row[0]) + 1
        window_end = int(row[1]) + 1
        #print("WINDOW: " + str(window_start)+" to " + str(window_end))
        normalization_factors = numpy.ones(no_bin)

        for _, gene in gene_spacer_counts_df.iterrows():
            gene_name = gene['Gene_Name']
            gene_start = min(int(gene['start']), int(gene['end']))
            gene_end = max(int(gene['start']), int(gene['end']))

            # Checking cases where an operon spans the whole window
            if (gene_start <= window_start) and (gene_end >= window_end): 
                #print("spanned the whole winodw: ")
                #print(gene_name)
                #print("gene start: " + str(gene_start))
                #print("gene end: " + str(gene_end))
                normalization_factors[:] = float(gene['spacer_count'])
                break
            
            # Checking cases where an operon starts before the window and ends within the window
            if (gene_start <= window_start) and (window_start  <= gene_end <= window_end):
                #print("ended in window:")
                #print(gene_name)
                #print("gene start: " + str(gene_start))
                #print("gene end: " + str(gene_end))
                end_index = int((gene_end - window_start)/binsize)
                normalization_factors[:end_index] = float(gene['spacer_count'])
            
            # Checking cases where an operon starts and ends within the window
            if (window_start  <= gene_start <= window_end) and (window_start  <= gene_end <= window_end):
                #print("started and ended in window:")
                #print(gene_name)
                #print("gene start: " + str(gene_start))
                #print("gene end: " + str(gene_end))
                start_index = int((gene_start - window_start)/binsize)
                end_index = int((gene_end - window_start)/binsize)
                normalization_factors[start_index:end_index] = float(gene['spacer_count'])
            
            # Checking cases where an operon starts within the window and ends after the window
            if (window_start  <= gene_start <= window_end) and (gene_end >= window_end):
                #print("started in window:")
                #print(gene_name)
                #print("gene start: " + str(gene_start))
                #print("gene end: " + str(gene_end))
                start_index = int((gene_start - window_start)/binsize)
                normalization_factors[start_index:] = float(gene['spacer_count'])
            
        # Normalizing regions between operons with the count of the closest operon
        #print("normalizataion factors before filling ones:")
        #print(normalization_factors)
        normalization_factors, status = replace_ones(normalization_factors)
        #print("normalizataion factors after filling ones:")
        #print(normalization_factors)
        if status == "only_ones":
            #print("DANGER DANGER DANGER ONLY ONEES ALAALALALALLAA")
            windows_to_delete.append(index)
        
        normalized_coverage_data[index,2:] = (10000 * coverage_data[index,2:]) / normalization_factors
        #print("coverage:")
        #print(coverage_data[index,:])
        #print("normalized coverage:")
        #print(normalized_coverage_data[index,:])

    normalized_coverage_data = numpy.delete(normalized_coverage_data, windows_to_delete, axis=0)
    normalized_coverage_data_no_window = normalized_coverage_data[:,2:]
    sequence_data = numpy.delete(sequence_data, windows_to_delete, axis=0)

    return normalized_coverage_data, normalized_coverage_data_no_window, sequence_data

def plot_window_coverage_binarized(binarized_coverage_with_windows_info, no_plots, no_bin, outpath, window_size, operon_df, gene_df, binsize, random = True):
    x_coord = numpy.arange(0, no_bin)               
    no_plots = int(no_plots)
    if random:
        indices = numpy.random.choice(binarized_coverage_with_windows_info.shape[0], no_plots, replace=False)
    else:
        total_entries = int(len(binarized_coverage_with_windows_info))
        indices = range(total_entries - no_plots, total_entries)
    counter = 0
    for idx in indices:
        counter += 1
        print(idx)
        print("fraction done: ")
        print(counter / no_plots)
        binarized_coverage = binarized_coverage_with_windows_info[idx, 2:]  # Skip the first two columns which contain the window start and end sites.
        window_start = int(binarized_coverage_with_windows_info[idx, 0])
        window_end = int(binarized_coverage_with_windows_info[idx, 1])

        gene_starts = []
        gene_ends = []
        operon_starts = []
        operon_ends = []

        for _, gene_row in gene_df.iterrows():
            if gene_row['leftEndPos'] == 'None' or gene_row['rightEndPos']  == 'None':
                continue
            
            gene_start = int(gene_row['leftEndPos']) - window_start +1 # + 1 because genome sequenced is indexed from 0 but gene and operon locations are indexed from 1
            gene_end = int(gene_row['rightEndPos']) - window_start +1 
            
            # Ensure correct ordering of start and end
            gene_start, gene_end = min(gene_start, gene_end), max(gene_start, gene_end)

            if 0 <= gene_start < window_size:
                #print("start of: " + gene_row['geneName']+ "is in this window")
                #print("gene starts at:" + str(gene_row['leftEndPos']))
                #print("gene ends at:" + str(gene_row['rightEndPos']))
                gene_starts.append(int(gene_start/binsize))
            if 0 <= gene_end < window_size:
                #print("end of: " + gene_row['geneName'] + "is in this window")
                #print("gene starts at:" + str(gene_row['leftEndPos']))
                #print("gene ends at:" + str(gene_row['rightEndPos']))
                gene_ends.append(int(gene_end/binsize))

        # Populate operon and directionality vectors
        for _, operon in operon_df.iterrows():
            operon_start = int(operon['firstGeneLeftPos']) - window_start +1 
            operon_end = int(operon['lastGeneRightPos']) - window_start +1 

            # Ensure correct ordering of start and end
            operon_start, operon_end = min(operon_start, operon_end), max(operon_start, operon_end)

            # start and end sites of operons are marked
            if (0 <= operon_start < window_size):
                operon_starts.append(int(operon_start/binsize))
                #print("start of: " + operon['operonName'] + "is in this window")
                #print("operon starts at:" + str(operon['firstGeneLeftPos']))
                #print("operon ends at:" + str(operon['lastGeneRightPos']))
#
            if (0 <= operon_end < window_size):
                operon_ends.append(int(operon_end/binsize))

        # Plotting
        pyplot.style.use('ggplot')
        pyplot.figure(figsize=(12, 6))
        ymax = binarized_coverage.max() + (binarized_coverage.max() * 0.1)
        pyplot.plot(x_coord, binarized_coverage, color="blue", label='Observed Coverage Normalized by Gene Expression binarized (peak yes/no)')
        pyplot.title(f"Normalized coverage over window: {window_start}-{window_end}")
        pyplot.xlabel('Bins')
        pyplot.ylabel('Normalized Coverage')
        pyplot.ylim(ymin=-0.2, ymax=ymax) 
        pyplot.xticks([0, round(no_bin/4), round(no_bin/2), round(3*no_bin/4), round(no_bin)], [0, round(no_bin/4), round(no_bin/2), round(3*no_bin/4), round(no_bin)])
        for gene_start in gene_starts:
            pyplot.axvline(x=gene_start, ls="-", lw="1.2",color="green")
            pyplot.text(gene_start + (no_bin/80), ymax.max() * 0.9, 'gene start', verticalalignment='center', color='green')
        for gene_end in gene_ends:
            pyplot.axvline(x=gene_end, ls="-", lw="1.2",color="green")
            pyplot.text(gene_end + (no_bin/80), ymax.max() * 0.3, 'gene end', verticalalignment='center', color='green')
        for operon_start in operon_starts:
            pyplot.axvline(x=operon_start, ls="-.", lw="1",color="red")
            pyplot.text(operon_start + (no_bin/80), ymax.max() * 0.8, 'operon start', verticalalignment='center', color='red')
        for operon_end in operon_ends:
            pyplot.axvline(x=operon_end, ls="-.", lw="1",color="red")
            pyplot.text(operon_end + (no_bin/80), ymax.max() * 0.2, 'operon end', verticalalignment='center', color='red')
        # Plot gene and operon bodies
        for _, gene_row in gene_df.iterrows():
            if gene_row['leftEndPos'] == 'None' or gene_row['rightEndPos']  == 'None':
                continue
            gene_start = int(gene_row['leftEndPos']) - window_start + 1
            gene_end = int(gene_row['rightEndPos']) - window_start + 1
            gene_start, gene_end = min(gene_start, gene_end), max(gene_start, gene_end)
            gene_name = gene_row['geneName']

            # Check if gene start or end is within the window
            if (0 <= gene_start < window_size) or (0 <= gene_end < window_size):
                # Adjust for window boundaries
                gene_start = max(0, gene_start)
                gene_end = min(window_size - 1, gene_end)
                gene_y = -(0.05 * ymax)
                pyplot.hlines(y=gene_y, xmin=gene_start/binsize, xmax=gene_end/binsize, colors='green', linestyles='solid')
                label_x_position = gene_start/binsize if gene_start >= 0 else gene_end/binsize
                pyplot.text(label_x_position, 1.5*gene_y, gene_name, color='green', fontsize=8)

        for _, operon_row in operon_df.iterrows():
            operon_start = int(operon_row['firstGeneLeftPos']) - window_start + 1
            operon_end = int(operon_row['lastGeneRightPos']) - window_start + 1
            operon_start, operon_end = min(operon_start, operon_end), max(operon_start, operon_end)
            operon_name = operon_row['operonName']

            # Check if operon start or end is within the window
            if (0 <= operon_start < window_size) or (0 <= operon_end < window_size) or ((operon_start < window_size) and (operon_end > window_size)):
                if ((operon_start < window_size) and (operon_end > window_size)):
                    pyplot.hlines(y=2*gene_y, xmin=0, xmax=binsize, colors='red', linestyles='solid')
                    label_x_position = 0.5 * binsize
                    pyplot.text(label_x_position, 2.5*gene_y, operon_name, color='red', fontsize=8)
                else:
                    operon_start = max(0, operon_start)
                    operon_end = min(window_size - 1, operon_end)

                    pyplot.hlines(y=2*gene_y, xmin=operon_start/binsize, xmax=operon_end/binsize, colors='red', linestyles='solid')
                    label_x_position = operon_start/binsize if operon_start >= 0 else operon_end/binsize
                    pyplot.text(label_x_position, 2.5*gene_y, operon_name, color='red', fontsize=8)
        pyplot.legend()
        pyplot.savefig(outpath+"_"+str(window_start)+"_"+str(window_end)+"_operoncoverage_normalized.png")
        print(outpath+"_"+str(window_start)+"_"+str(window_end)+"_operoncoverage_normalized.png")
        pyplot.close()
    return "Plots generated"

def plot_window_coverage_normalized(normalized_coverage_with_windows_info, no_plots, no_bin, outpath, window_size, operon_df, gene_df, binsize, random = True):
    x_coord = numpy.arange(0, no_bin)               
    no_plots = int(no_plots)
    if random:
        indices = numpy.random.choice(normalized_coverage_with_windows_info.shape[0], no_plots, replace=False)
    else:
        total_entries = int(len(normalized_coverage_with_windows_info))
        indices = range(total_entries - no_plots, total_entries)
    counter = 0
    for idx in indices:
        counter += 1
        print(idx)
        print("fraction done: ")
        print(counter / no_plots)
        print(normalized_coverage_with_windows_info[idx, :10])
        normalized_coverage = normalized_coverage_with_windows_info[idx, 2:]  # Skip the first two columns which contain the window start and end sites.
        window_start = int(normalized_coverage_with_windows_info[idx, 0])
        window_end = int(normalized_coverage_with_windows_info[idx, 1])

        gene_starts = []
        gene_ends = []
        operon_starts = []
        operon_ends = []

        for _, gene_row in gene_df.iterrows():
            if gene_row['leftEndPos'] == 'None' or gene_row['rightEndPos']  == 'None':
                continue
            
            gene_start = int(gene_row['leftEndPos']) - window_start +1 # + 1 because genome sequenced is indexed from 0 but gene and operon locations are indexed from 1
            gene_end = int(gene_row['rightEndPos']) - window_start +1 
            
            # Ensure correct ordering of start and end
            gene_start, gene_end = min(gene_start, gene_end), max(gene_start, gene_end)

            if 0 <= gene_start < window_size:
                #print("start of: " + gene_row['geneName']+ "is in this window")
                #print("gene starts at:" + str(gene_row['leftEndPos']))
                #print("gene ends at:" + str(gene_row['rightEndPos']))
                gene_starts.append(int(gene_start/binsize))
            if 0 <= gene_end < window_size:
                #print("end of: " + gene_row['geneName'] + "is in this window")
                #print("gene starts at:" + str(gene_row['leftEndPos']))
                #print("gene ends at:" + str(gene_row['rightEndPos']))
                gene_ends.append(int(gene_end/binsize))

        # Populate operon and directionality vectors
        for _, operon in operon_df.iterrows():
            operon_start = int(operon['firstGeneLeftPos']) - window_start +1 
            operon_end = int(operon['lastGeneRightPos']) - window_start +1 

            # Ensure correct ordering of start and end
            operon_start, operon_end = min(operon_start, operon_end), max(operon_start, operon_end)

            # start and end sites of operons are marked
            if (0 <= operon_start < window_size):
                operon_starts.append(int(operon_start/binsize))
                #print("start of: " + operon['operonName'] + "is in this window")
                #print("operon starts at:" + str(operon['firstGeneLeftPos']))
                #print("operon ends at:" + str(operon['lastGeneRightPos']))
#
            if (0 <= operon_end < window_size):
                operon_ends.append(int(operon_end/binsize))

        # Plotting
        pyplot.style.use('ggplot')
        pyplot.figure(figsize=(12, 6))
        ymax = normalized_coverage.max() + (normalized_coverage.max() * 0.1)
        pyplot.plot(x_coord, normalized_coverage, color="blue", label='Observed Coverage Normalized by Gene Expression')
        pyplot.title(f"Normalized coverage over window: {window_start}-{window_end}")
        pyplot.xlabel('Bins')
        pyplot.ylabel('Normalized Coverage')
        pyplot.ylim(ymin=-0.2, ymax=ymax) 
        pyplot.xticks([0, round(no_bin/4), round(no_bin/2), round(3*no_bin/4), round(no_bin)], [0, round(no_bin/4), round(no_bin/2), round(3*no_bin/4), round(no_bin)])
        for gene_start in gene_starts:
            pyplot.axvline(x=gene_start, ls="-", lw="1.2",color="green")
            pyplot.text(gene_start + (no_bin/80), ymax.max() * 0.9, 'gene start', verticalalignment='center', color='green')
        for gene_end in gene_ends:
            pyplot.axvline(x=gene_end, ls="-", lw="1.2",color="green")
            pyplot.text(gene_end + (no_bin/80), ymax.max() * 0.3, 'gene end', verticalalignment='center', color='green')
        for operon_start in operon_starts:
            pyplot.axvline(x=operon_start, ls="-.", lw="1",color="red")
            pyplot.text(operon_start + (no_bin/80), ymax.max() * 0.8, 'operon start', verticalalignment='center', color='red')
        for operon_end in operon_ends:
            pyplot.axvline(x=operon_end, ls="-.", lw="1",color="red")
            pyplot.text(operon_end + (no_bin/80), ymax.max() * 0.2, 'operon end', verticalalignment='center', color='red')
        # Plot gene and operon bodies
        for _, gene_row in gene_df.iterrows():
            if gene_row['leftEndPos'] == 'None' or gene_row['rightEndPos']  == 'None':
                continue
            gene_start = int(gene_row['leftEndPos']) - window_start + 1
            gene_end = int(gene_row['rightEndPos']) - window_start + 1
            gene_start, gene_end = min(gene_start, gene_end), max(gene_start, gene_end)
            gene_name = gene_row['geneName']

            # Check if gene start or end is within the window
            if (0 <= gene_start < window_size) or (0 <= gene_end < window_size):
                # Adjust for window boundaries
                gene_start = max(0, gene_start)
                gene_end = min(window_size - 1, gene_end)
                gene_y = -(0.05 * ymax)
                pyplot.hlines(y=gene_y, xmin=gene_start/binsize, xmax=gene_end/binsize, colors='green', linestyles='solid')
                label_x_position = gene_start/binsize if gene_start >= 0 else gene_end/binsize
                pyplot.text(label_x_position, 1.5*gene_y, gene_name, color='green', fontsize=8)

        for _, operon_row in operon_df.iterrows():
            operon_start = int(operon_row['firstGeneLeftPos']) - window_start + 1
            operon_end = int(operon_row['lastGeneRightPos']) - window_start + 1
            operon_start, operon_end = min(operon_start, operon_end), max(operon_start, operon_end)
            operon_name = operon_row['operonName']

            # Check if operon start or end is within the window
            if (0 <= operon_start < window_size) or (0 <= operon_end < window_size) or ((operon_start < window_size) and (operon_end > window_size)):
                if ((operon_start < window_size) and (operon_end > window_size)):
                    pyplot.hlines(y=2*gene_y, xmin=0, xmax=binsize, colors='red', linestyles='solid')
                    label_x_position = 0.5 * binsize
                    pyplot.text(label_x_position, 2.5*gene_y, operon_name, color='red', fontsize=8)
                else:
                    operon_start = max(0, operon_start)
                    operon_end = min(window_size - 1, operon_end)

                    pyplot.hlines(y=2*gene_y, xmin=operon_start/binsize, xmax=operon_end/binsize, colors='red', linestyles='solid')
                    label_x_position = operon_start/binsize if operon_start >= 0 else operon_end/binsize
                    pyplot.text(label_x_position, 2.5*gene_y, operon_name, color='red', fontsize=8)
        pyplot.legend()
        pyplot.savefig(outpath+"_"+str(window_start)+"_"+str(window_end)+"_operoncoverage_normalized.png")
        print(outpath+"_"+str(window_start)+"_"+str(window_end)+"_operoncoverage_normalized.png")
        pyplot.close()
    return "Plots generated"


def custom_train_test_split(X, Y, window_size, overlap, test_size,  random_state=None):
    
    if random_state is not None:
        numpy.random.seed(random_state)
    # Calculate the number of samples
    n_samples = X.shape[0]

    # Calculate the number of samples to be used in the test set
    n_test_samples = int(n_samples * test_size)

    # Calculate the safety margin to avoid overlap
    # The safety margin is the number of windows needed to cover the overlap
    safety_margin = int(overlap / (window_size - overlap))

    # Calculate the start index for the test set
    test_start_index = n_samples - n_test_samples - safety_margin

    # Split the data into training and testing sets
    X_train = X[:test_start_index]
    Y_train = Y[:test_start_index]
    X_test = X[test_start_index + safety_margin:]
    Y_test = Y[test_start_index + safety_margin:]

    X_train, Y_train = shuffle(X_train, Y_train, random_state=random_state)

    return X_train, X_test, Y_train, Y_test
        

   