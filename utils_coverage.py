import numpy 
import pandas as pd
import HTSeq
from matplotlib import pyplot

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
        for bam_file in counts_df.columns[6:]:
            spacer_count += row[bam_file]
        gene_dict[gene_name] = spacer_count
    
    sorted_genes = sorted(gene_dict, key=lambda k: gene_dict[k])
    number_to_be_dropped = int(len(sorted_genes) * (gene_perc / 100))
    return sorted_genes[:number_to_be_dropped]

def lowest_expressed_genes_per_bamlist(bamlist, counts_df, gene_perc):   # not used
    gene_dict = {}
    for index, row in counts_df.iterrows():
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
            coverage_array = coverage_array / count_dict[bam_file]  # Normalize by total number of spacers per bam file
            coverage_array, num_bins = bin_coverage(coverage_array, binsize)  # Bin the coverage array
            coverage_row = [window_start, window_end] + coverage_array.tolist()
            batch_coverage_list.append(coverage_row)

    # Convert to DataFrame
    batch_coverage_columns = ['Window_Start', 'Window_End'] + [f'Pos_{i}' for i in range(num_bins)]
    batch_coverage_df = pd.DataFrame(batch_coverage_list, columns=batch_coverage_columns)
    return batch_coverage_df

def plot_window_coverage(df, window_size, operon_df, gene_df, plot_no, no_bin, binsize, outpath):

    windows_to_plot = df.sample(plot_no)

    counter = 0
    for _, row in windows_to_plot.iterrows():
        counter += 1
        print("fraction done: ")
        print(counter / plot_no)
        window_start, window_end = int(row['Window_Start']), int(row['Window_End'])

        window_start = int(window_start)
        window_end = int(window_end)
        coverage = row["Pos_0":]
        coverage = coverage.to_numpy()


        gene_starts = []
        gene_ends = []
        operon_starts = []
        operon_ends = []
        print("window from: "+ str(window_start) + " to: " + str(window_end))
        print("-------GENES------")
        for _, gene_row in gene_df.iterrows():
            if gene_row['leftEndPos'] == 'None' or gene_row['rightEndPos']  == 'None':
                continue
            
            gene_start = int(gene_row['leftEndPos']) - window_start -1 # -1 because genome sequenced is indexed from 0 but gene and operon locations are indexed from 1
            gene_end = int(gene_row['rightEndPos']) - window_start -1 
            
            # Ensure correct ordering of start and end
            gene_start, gene_end = min(gene_start, gene_end), max(gene_start, gene_end)

            if 0 <= gene_start < window_size:
                print("start of: " + gene_row['geneName']+ "is in this window")
                print("gene starts at:" + str(gene_row['leftEndPos']))
                print("gene ends at:" + str(gene_row['rightEndPos']))
                gene_starts.append(int(gene_start/binsize))
            if 0 <= gene_end < window_size:
                print("end of: " + gene_row['geneName'] + "is in this window")
                print("gene starts at:" + str(gene_row['leftEndPos']))
                print("gene ends at:" + str(gene_row['rightEndPos']))
                gene_ends.append(int(gene_end/binsize))
        print("-------OPERONS------")
        # Populate operon and directionality vectors
        for _, operon in operon_df.iterrows():
            operon_start = int(operon['firstGeneLeftPos']) - window_start -1 
            operon_end = int(operon['lastGeneRightPos']) - window_start -1 

            # Ensure correct ordering of start and end
            operon_start, operon_end = min(operon_start, operon_end), max(operon_start, operon_end)

            # start and end sites of operons are marked
            if (0 <= operon_start < window_size):
                operon_starts.append(int(operon_start/binsize))
                print("start of: " + operon['operonName'] + "is in this window")
                print("operon starts at:" + str(operon['firstGeneLeftPos']))
                print("operon ends at:" + str(operon['lastGeneRightPos']))

            if (0 <= operon_end < window_size):
                operon_ends.append(int(operon_end/binsize))
                print("end of: " + operon['operonName'] + "is in this window")
                print("operon starts at:" + str(operon['firstGeneLeftPos']))
                print("operon ends at:" + str(operon['lastGeneRightPos']))
        x_coord=numpy.arange(0, no_bin)
        
        pyplot.style.use('ggplot')
        pyplot.figure(figsize=(12, 6))
        pyplot.plot( x_coord, coverage, color="blue")
        pyplot.title("Window from genome position " + str(window_start) + " to " + str(window_end))
        pyplot.axvline(x=0, ls="-", lw="2", color = 'blue')
        pyplot.axvline(x=no_bin, ls="-", lw="2", color = 'blue')
        for gene_start in gene_starts:
            pyplot.axvline(x=gene_start, ls="-.", lw="1",color="green")
            pyplot.text(gene_start + (no_bin/40), coverage.max() * 0.9, 'gene start', verticalalignment='center', color='green')
        for gene_end in gene_ends:
            pyplot.axvline(x=gene_end, ls="-.", lw="1",color="green")
            pyplot.text(gene_end + (no_bin/40), coverage.max() * 0.8, 'gene end', verticalalignment='center', color='green')
        for operon_start in operon_starts:
            pyplot.axvline(x=operon_start, ls="-.", lw="1",color="red")
            pyplot.text(operon_start + (no_bin/40), coverage.max() * 0.7, 'operon start', verticalalignment='center', color='red')
        for operon_end in operon_ends:
            pyplot.axvline(x=operon_end, ls="-.", lw="1",color="red")
            pyplot.text(operon_end + (no_bin/40), coverage.max() * 0.6, 'operon end', verticalalignment='center', color='red')
        pyplot.xlabel('Bins')
        pyplot.ylabel('Normalized coverage summed over bam files')
        pyplot.ylim(ymin = 0, ymax=coverage.max()+(coverage.max()*0.1)) 
        pyplot.xticks([0, round(no_bin/4), round(no_bin/2), round(3*no_bin/4), round(no_bin)], [0, round(no_bin/4), round(no_bin/2), round(3*no_bin/4), round(no_bin)])
        pyplot.savefig(outpath+"_"+str(window_start)+"_"+str(window_end)+"_operoncoverage.png")
        pyplot.close()

    return "plots generated"

            
        

   