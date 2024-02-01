import numpy
from matplotlib import pyplot





def plot_window_coverage_normalized(normalized_coverage_with_windows_info, no_plots, no_bin, outpath, dataset_name, window_size, promoter_df, terminator_df, gene_df, binsize, random = True):
    outpath = outpath + dataset_name+ "_outputs" + "/" + "window_coverage_plots/"
    
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
        
        normalized_coverage = normalized_coverage_with_windows_info[idx, 2:]  # Skip the first two columns which contain the window start and end sites.
        window_start = int(normalized_coverage_with_windows_info[idx, 0])
        window_end = int(normalized_coverage_with_windows_info[idx, 1])

        gene_starts = []
        gene_ends = []
        terminator_starts = []
        terminator_ends = []
        promoters = []

        for _, gene_row in gene_df.iterrows():
            if gene_row['Left'] == 'None' or gene_row['Right']  == 'None':
                continue
            
            gene_start = int(gene_row['Left']) - window_start
            gene_end = int(gene_row['Right']) - window_start 
            
            # Ensure correct ordering of start and end
            gene_start, gene_end = min(gene_start, gene_end), max(gene_start, gene_end)

            if 0 <= gene_start < window_size:
                print("start of: " + gene_row['Gene_Name']+ "is in this window")
                print("gene starts at:" + str(gene_row['Left']))
                print("gene ends at:" + str(gene_row['Right']))
                gene_starts.append(int(gene_start/binsize))
            if 0 <= gene_end < window_size:
                print("end of: " + gene_row['Gene_Name'] + "is in this window")
                print("gene starts at:" + str(gene_row['Left']))
                print("gene ends at:" + str(gene_row['Right']))
                gene_ends.append(int(gene_end/binsize))
        
        for _, promoter_row in promoter_df.iterrows():
            
            promoter_position = int(promoter_row['Absolute_Plus_1_Position']) - window_start
            
            if 0 <= promoter_position < window_size:
                promoters.append(int(promoter_position/binsize))

        for _, terminator in terminator_df.iterrows():
            terminator_start = int(terminator['Left_End_Position']) - window_start 
            terminator_end = int(terminator['Right_End_Position']) - window_start 

            # Ensure correct ordering of start and end
            terminator_start, terminator_end = min(terminator_start, terminator_end), max(terminator_start, terminator_end)

            if (0 <= terminator_start < window_size):
                terminator_starts.append(int(terminator_start/binsize))
                
            if (0 <= terminator_end < window_size):
                terminator_ends.append(int(terminator_end/binsize))

        # Plotting
        pyplot.style.use('ggplot')
        pyplot.figure(figsize=(12, 6))
        ymax = normalized_coverage.max() + (normalized_coverage.max() * 0.1)
        pyplot.plot(x_coord, normalized_coverage, color="blue", label='Observed Coverage Normalized by Gene Expression')
        pyplot.title(f"Normalized coverage over window: {window_start}-{window_end}")
        pyplot.xlabel('Bins')
        pyplot.ylabel('Normalized Coverage')
        pyplot.ylim(ymin=-0.2*ymax, ymax=ymax) 
        pyplot.xticks([0, round(no_bin/4), round(no_bin/2), round(3*no_bin/4), round(no_bin)], [0, round(no_bin/4), round(no_bin/2), round(3*no_bin/4), round(no_bin)])
        for gene_start in gene_starts:
            pyplot.axvline(x=gene_start, ls="-", lw="1.2",color="green")
            pyplot.text(gene_start + (no_bin/80), ymax.max() * 0.9, 'gene start', verticalalignment='center', color='green')
        for gene_end in gene_ends:
            pyplot.axvline(x=gene_end, ls="-", lw="1.2",color="green")
            pyplot.text(gene_end + (no_bin/80), ymax.max() * 0.7, 'gene end', verticalalignment='center', color='green')
        for promoter in promoters:
            pyplot.axvline(x=promoter, ls="-.", lw="1.2",color="brown")
            pyplot.text(promoter + (no_bin/80), ymax.max() * 0.2, 'promoter', verticalalignment='center', color='brown')
        
        # Plot gene bodies and terminators
        for _, gene_row in gene_df.iterrows():
            if gene_row['Left'] == 'None' or gene_row['Right']  == 'None':
                continue
            gene_start = int(gene_row['Left']) - window_start 
            gene_end = int(gene_row['Right']) - window_start 
            gene_start, gene_end = min(gene_start, gene_end), max(gene_start, gene_end)
            gene_name = gene_row['Gene_Name']

            # Check if gene start or end is within the window
            if (0 <= gene_start < window_size) or (0 <= gene_end < window_size):
                # Adjust for window boundaries
                gene_start = max(0, gene_start)
                gene_end = min(window_size - 1, gene_end)
                gene_y = -(0.05 * ymax)
                pyplot.hlines(y=gene_y, xmin=gene_start/binsize, xmax=gene_end/binsize, colors='green', linestyles='solid')
                label_x_position = gene_start/binsize if gene_start >= 0 else gene_end/binsize
                pyplot.text(label_x_position, 1.5*gene_y, gene_name, color='green', fontsize=8)

        for _, terminator_row in terminator_df.iterrows():
            terminator_start = int(terminator_row['Left_End_Position']) - window_start 
            terminator_end = int(terminator_row['Right_End_Position']) - window_start 
            terminator_start, terminator_end = min(terminator_start, terminator_end), max(terminator_start, terminator_end)
            terminator_name = terminator_row['Terminator_ID']

            # Check if terminator start or end is within the window
            if (0 <= terminator_start < window_size) or (0 <= terminator_end < window_size) or ((terminator_start < window_size) and (terminator_end > window_size)):
                if ((terminator_start < window_size) and (terminator_end > window_size)):
                    pyplot.hlines(y=2*gene_y, xmin=0, xmax=binsize, colors='red', linestyles='solid')
                    label_x_position = 0.5 * binsize
                    pyplot.text(label_x_position, 2.5*gene_y, terminator_name, color='red', fontsize=8)
                else:
                    terminator_start = max(0, terminator_start)
                    terminator_end = min(window_size - 1, terminator_end)

                    pyplot.hlines(y=2*gene_y, xmin=terminator_start/binsize, xmax=terminator_end/binsize, colors='red', linestyles='solid')
                    label_x_position = terminator_start/binsize if terminator_start >= 0 else terminator_end/binsize
                    pyplot.text(label_x_position, 2.5*gene_y, terminator_name, color='red', fontsize=8)
        pyplot.legend()
        pyplot.savefig(outpath+"_"+str(window_start)+"_"+str(window_end)+"_window_coverage.png")
        print(outpath+"_"+str(window_start)+"_"+str(window_end)+"_window_coverage.png")
        pyplot.close()
    return "Plots generated"



def plot_predicted_vs_observed(model, model_name, X_test_seq, X_test_annot, normalized_coverage_with_windows_info, no_plots, no_bin, outpath, dataset_name, window_size, promoter_df, terminator_df, gene_df, binsize, log_scale):
    outpath = outpath + dataset_name+ "_outputs" + "/" + "prediction_plots/"
    
    x_coord = numpy.arange(0, no_bin)               
    no_plots = int(no_plots)
    
    indices = numpy.random.choice(X_test_seq.shape[0], no_plots, replace=False)
    counter = 0
    for idx in indices:
        counter += 1
        print("fraction done: ")
        print(counter / no_plots)

        coverage_predicted = model.predict([X_test_seq[idx:idx+1], X_test_annot[idx:idx+1]])[0]
        
        normalized_coverage = normalized_coverage_with_windows_info[idx, 2:]  # Skip the first two columns which contain the window start and end sites.

        window_start = int(normalized_coverage_with_windows_info[idx, 0])
        window_end = int(normalized_coverage_with_windows_info[idx, 1])
        gene_starts = []
        gene_ends = []
        terminator_starts = []
        terminator_ends = []
        promoters = []

        for _, gene_row in gene_df.iterrows():
            if gene_row['Left'] == 'None' or gene_row['Right']  == 'None':
                continue
            
            gene_start = int(gene_row['Left']) - window_start
            gene_end = int(gene_row['Right']) - window_start 
            
            # Ensure correct ordering of start and end
            gene_start, gene_end = min(gene_start, gene_end), max(gene_start, gene_end)

            if 0 <= gene_start < window_size:
                gene_starts.append(int(gene_start/binsize))
            if 0 <= gene_end < window_size:
                gene_ends.append(int(gene_end/binsize))
        
        for _, promoter_row in promoter_df.iterrows():
            
            promoter_position = int(promoter_row['Absolute_Plus_1_Position']) - window_start
            
            if 0 <= promoter_position < window_size:
                promoters.append(int(promoter_position/binsize))

        for _, terminator in terminator_df.iterrows():
            terminator_start = int(terminator['Left_End_Position']) - window_start 
            terminator_end = int(terminator['Right_End_Position']) - window_start 

            # Ensure correct ordering of start and end
            terminator_start, terminator_end = min(terminator_start, terminator_end), max(terminator_start, terminator_end)

            if (0 <= terminator_start < window_size):
                terminator_starts.append(int(terminator_start/binsize))
         
#
            if (0 <= terminator_end < window_size):
                terminator_ends.append(int(terminator_end/binsize))

        # Plotting
        pyplot.style.use('ggplot')
        pyplot.figure(figsize=(12, 6))
        ymax = normalized_coverage.max() + (normalized_coverage.max() * 0.1)
        pyplot.plot(x_coord, normalized_coverage, color="blue", label='Observed Coverage Normalized by Gene Expression')
        pyplot.plot(x_coord, coverage_predicted, color="purple", label='Predicted Coverage')
        pyplot.title(f"Predicted vs. Observed Coverage over Window {window_start}-{window_end}")
        pyplot.xlabel('Bins')
        pyplot.ylabel('Normalized Coverage')
        if log_scale == True:
            pyplot.ylabel('Log10(Normalized Coverage)')
            pyplot.yscale('log')
        pyplot.ylim(ymin=-0.2*ymax, ymax=ymax) 
        pyplot.xticks([0, round(no_bin/4), round(no_bin/2), round(3*no_bin/4), round(no_bin)], [0, round(no_bin/4), round(no_bin/2), round(3*no_bin/4), round(no_bin)])
        #for gene_start in gene_starts:
        #    pyplot.axvline(x=gene_start, ls="-", lw="1.2",color="green")
        #    pyplot.text(gene_start + (no_bin/80), ymax.max() * 0.9, 'gene start', verticalalignment='center', color='green')
        #for gene_end in gene_ends:
        #    pyplot.axvline(x=gene_end, ls="-", lw="1.2",color="green")
        #    pyplot.text(gene_end + (no_bin/80), ymax.max() * 0.7, 'gene end', verticalalignment='center', color='green')
        #for promoter in promoters:
        #    pyplot.axvline(x=promoter, ls="-.", lw="1.2",color="brown")
        #    pyplot.text(promoter + (no_bin/80), ymax.max() * 0.2, 'promoter', verticalalignment='center', color='brown')
        
        # Plot gene bodies and terminators
        for _, gene_row in gene_df.iterrows():
            if gene_row['Left'] == 'None' or gene_row['Right']  == 'None':
                continue
            gene_start = int(gene_row['Left']) - window_start 
            gene_end = int(gene_row['Right']) - window_start 
            gene_start, gene_end = min(gene_start, gene_end), max(gene_start, gene_end)
            gene_name = gene_row['Gene_Name']

            # Check if gene start or end is within the window
            if (0 <= gene_start < window_size) or (0 <= gene_end < window_size):
                # Adjust for window boundaries
                gene_start = max(0, gene_start)
                gene_end = min(window_size - 1, gene_end)
                gene_y = -(0.05 * ymax)
                pyplot.hlines(y=gene_y, xmin=gene_start/binsize, xmax=gene_end/binsize, colors='green', linestyles='solid')
                label_x_position = gene_start/binsize if gene_start >= 0 else gene_end/binsize
                pyplot.text(label_x_position, 1.5*gene_y, gene_name, color='green', fontsize=8)

        for _, terminator_row in terminator_df.iterrows():
            terminator_start = int(terminator_row['Left_End_Position']) - window_start 
            terminator_end = int(terminator_row['Right_End_Position']) - window_start 
            terminator_start, terminator_end = min(terminator_start, terminator_end), max(terminator_start, terminator_end)
            terminator_name = terminator_row['Terminator_ID']

            # Check if terminator start or end is within the window
            if (0 <= terminator_start < window_size) or (0 <= terminator_end < window_size) or ((terminator_start < window_size) and (terminator_end > window_size)):
                if ((terminator_start < window_size) and (terminator_end > window_size)):
                    pyplot.hlines(y=2*gene_y, xmin=0, xmax=binsize, colors='red', linestyles='solid')
                    label_x_position = 0.5 * binsize
                    pyplot.text(label_x_position, 2.5*gene_y, terminator_name, color='red', fontsize=8)
                else:
                    terminator_start = max(0, terminator_start)
                    terminator_end = min(window_size - 1, terminator_end)

                    pyplot.hlines(y=2*gene_y, xmin=terminator_start/binsize, xmax=terminator_end/binsize, colors='red', linestyles='solid')
                    label_x_position = terminator_start/binsize if terminator_start >= 0 else terminator_end/binsize
                    pyplot.text(label_x_position, 2.5*gene_y, terminator_name, color='red', fontsize=8)
        pyplot.legend()
        if log_scale == True:
            pyplot.savefig(outpath+str(window_start)+"_"+str(window_end)+ model_name + "predicted_coverage_log.png")
        else:
            pyplot.savefig(outpath+str(window_start)+"_"+str(window_end)+ model_name + "predicted_coverage.png")
        pyplot.close()
    return "Plots generated"


def plot_window_coverage_normalized_compare_profiles(normalized_coverage_with_windows_info_1, normalized_coverage_with_windows_info_2, normalized_coverage_with_windows_info_3, normalized_coverage_with_windows_info_4, experiment_1_name, experiment_2_name , experiment_3_name, experiment_4_name , no_plots, no_bin, outpath, window_size, promoter_df, terminator_df, gene_df, binsize, random = True):
    outpath = outpath + "window_coverage_plots_experiment_comparison/"
    
    x_coord = numpy.arange(0, no_bin)               
    no_plots = int(no_plots)
    if random:
        indices = numpy.random.choice(normalized_coverage_with_windows_info_1.shape[0], no_plots, replace=False)
    else:
        total_entries = int(len(normalized_coverage_with_windows_info_1))
        indices = range(total_entries - no_plots, total_entries)
    counter = 0
    for idx in indices:
        counter += 1
        print(idx)
        print("fraction done: ")
        print(counter / no_plots)
        
        normalized_coverage_1 = normalized_coverage_with_windows_info_1[idx, 2:]  # Skip window start/end columns
        print("max coverage 1: ", normalized_coverage_1.max())
        window_start = int(normalized_coverage_with_windows_info_1[idx, 0])
        window_end = int(normalized_coverage_with_windows_info_1[idx, 1])

        matching_index_2 = numpy.where(
            (normalized_coverage_with_windows_info_2[:, 0] == window_start) &
            (normalized_coverage_with_windows_info_2[:, 1] == window_end)
        )[0]

        if matching_index_2.size == 0:
            continue

        # Use the first element of matching_index to get a 1D array
        normalized_coverage_2 = normalized_coverage_with_windows_info_2[matching_index_2[0], 2:]

        matching_index_3 = numpy.where(
            (normalized_coverage_with_windows_info_3[:, 0] == window_start) &
            (normalized_coverage_with_windows_info_3[:, 1] == window_end)
        )[0]

        if matching_index_3.size == 0:
            continue

        # Use the first element of matching_index to get a 1D array
        normalized_coverage_3 = normalized_coverage_with_windows_info_3[matching_index_3[0], 2:]

        matching_index_4 = numpy.where(
            (normalized_coverage_with_windows_info_4[:, 0] == window_start) &
            (normalized_coverage_with_windows_info_4[:, 1] == window_end)
        )[0]

        if matching_index_4.size == 0:
            continue

        # Use the first element of matching_index to get a 1D array
        normalized_coverage_4 = normalized_coverage_with_windows_info_4[matching_index_4[0], 2:]
        gene_starts = []
        gene_ends = []
        terminator_starts = []
        terminator_ends = []
        promoters = []

        for _, gene_row in gene_df.iterrows():
            if gene_row['Left'] == 'None' or gene_row['Right']  == 'None':
                continue
            
            gene_start = int(gene_row['Left']) - window_start
            gene_end = int(gene_row['Right']) - window_start 
            
            # Ensure correct ordering of start and end
            gene_start, gene_end = min(gene_start, gene_end), max(gene_start, gene_end)

            if 0 <= gene_start < window_size:
                #print("start of: " + gene_row['Gene_Name']+ "is in this window")
                #print("gene starts at:" + str(gene_row['Left']))
                #print("gene ends at:" + str(gene_row['Right']))
                gene_starts.append(int(gene_start/binsize))
            if 0 <= gene_end < window_size:
                #print("end of: " + gene_row['Gene_Name'] + "is in this window")
                #print("gene starts at:" + str(gene_row['Left']))
                #print("gene ends at:" + str(gene_row['Right']))
                gene_ends.append(int(gene_end/binsize))
        
        for _, promoter_row in promoter_df.iterrows():
            
            promoter_position = int(promoter_row['Absolute_Plus_1_Position']) - window_start
            
            if 0 <= promoter_position < window_size:
                promoters.append(int(promoter_position/binsize))

        for _, terminator in terminator_df.iterrows():
            terminator_start = int(terminator['Left_End_Position']) - window_start 
            terminator_end = int(terminator['Right_End_Position']) - window_start 

            # Ensure correct ordering of start and end
            terminator_start, terminator_end = min(terminator_start, terminator_end), max(terminator_start, terminator_end)

            if (0 <= terminator_start < window_size):
                terminator_starts.append(int(terminator_start/binsize))
                
            if (0 <= terminator_end < window_size):
                terminator_ends.append(int(terminator_end/binsize))

        # Plotting¨
        ymax = max(normalized_coverage_1.max() * 1.1, 
           normalized_coverage_2.max() * 1.1, 
           normalized_coverage_3.max() * 1.1, 
           normalized_coverage_4.max() * 1.1)
        pyplot.style.use('ggplot')
        pyplot.figure(figsize=(12, 6))
        pyplot.plot(x_coord, normalized_coverage_1, color="blue", linestyle='dashed', label=experiment_1_name)
        pyplot.plot(x_coord, normalized_coverage_2, color="blue", linestyle='dotted', label=experiment_2_name)
        pyplot.plot(x_coord, normalized_coverage_3, color="blue", linestyle='dashdot', label=experiment_3_name)
        pyplot.plot(x_coord, normalized_coverage_4, color="orange", linestyle='solid', label=experiment_4_name)
        pyplot.title(f"Normalized coverage over window: {window_start}-{window_end}")
        pyplot.xlabel('Bins')
        pyplot.ylabel('Spacer Coverage normalized for library size and gene expression')
        pyplot.ylim(ymin=-0.2*ymax, ymax=ymax) 
        pyplot.xticks([0, round(no_bin/4), round(no_bin/2), round(3*no_bin/4), round(no_bin)], [0, round(no_bin/4), round(no_bin/2), round(3*no_bin/4), round(no_bin)])
        for gene_start in gene_starts:
            pyplot.axvline(x=gene_start, ls="-", lw="1.2",color="green")
            pyplot.text(gene_start + (no_bin/80), ymax.max() * 0.9, 'gene start', verticalalignment='center', color='green')
        for gene_end in gene_ends:
            pyplot.axvline(x=gene_end, ls="-", lw="1.2",color="green")
            pyplot.text(gene_end + (no_bin/80), ymax.max() * 0.7, 'gene end', verticalalignment='center', color='green')
        for promoter in promoters:
            pyplot.axvline(x=promoter, ls="-.", lw="1.2",color="brown")
            pyplot.text(promoter + (no_bin/80), ymax.max() * 0.2, 'promoter', verticalalignment='center', color='brown')
        
        # Plot gene bodies and terminators
        for _, gene_row in gene_df.iterrows():
            if gene_row['Left'] == 'None' or gene_row['Right']  == 'None':
                continue
            gene_start = int(gene_row['Left']) - window_start 
            gene_end = int(gene_row['Right']) - window_start 
            gene_start, gene_end = min(gene_start, gene_end), max(gene_start, gene_end)
            gene_name = gene_row['Gene_Name']

            # Check if gene start or end is within the window
            if (0 <= gene_start < window_size) or (0 <= gene_end < window_size):
                # Adjust for window boundaries
                gene_start = max(0, gene_start)
                gene_end = min(window_size - 1, gene_end)
                gene_y = -(0.05 * ymax)
                pyplot.hlines(y=gene_y, xmin=gene_start/binsize, xmax=gene_end/binsize, colors='green', linestyles='solid')
                label_x_position = gene_start/binsize if gene_start >= 0 else gene_end/binsize
                pyplot.text(label_x_position, 1.5*gene_y, gene_name, color='green', fontsize=8)

        for _, terminator_row in terminator_df.iterrows():
            terminator_start = int(terminator_row['Left_End_Position']) - window_start 
            terminator_end = int(terminator_row['Right_End_Position']) - window_start 
            terminator_start, terminator_end = min(terminator_start, terminator_end), max(terminator_start, terminator_end)
            terminator_name = terminator_row['Terminator_ID']

            # Check if terminator start or end is within the window
            if (0 <= terminator_start < window_size) or (0 <= terminator_end < window_size) or ((terminator_start < window_size) and (terminator_end > window_size)):
                if ((terminator_start < window_size) and (terminator_end > window_size)):
                    pyplot.hlines(y=2*gene_y, xmin=0, xmax=binsize, colors='red', linestyles='solid')
                    label_x_position = 0.5 * binsize
                    pyplot.text(label_x_position, 2.5*gene_y, terminator_name, color='red', fontsize=8)
                else:
                    terminator_start = max(0, terminator_start)
                    terminator_end = min(window_size - 1, terminator_end)

                    pyplot.hlines(y=2*gene_y, xmin=terminator_start/binsize, xmax=terminator_end/binsize, colors='red', linestyles='solid')
                    label_x_position = terminator_start/binsize if terminator_start >= 0 else terminator_end/binsize
                    pyplot.text(label_x_position, 2.5*gene_y, terminator_name, color='red', fontsize=8)
        pyplot.legend()
        pyplot.savefig(outpath+"_"+str(window_start)+"_"+str(window_end)+"_window_coverage_" + experiment_4_name +".png")
        pyplot.close()
    return "Plots generated"