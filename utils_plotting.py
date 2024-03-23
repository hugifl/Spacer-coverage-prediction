import numpy
from matplotlib import pyplot
import os
from scipy.signal import find_peaks
import tensorflow as tf



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
    outpath = outpath + dataset_name+ "_outputs" + "/" + f"prediction_plots_{model_name}/"
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    x_coord = numpy.arange(0, no_bin)               
    no_plots = int(no_plots)
    numpy.random.seed(42) 
    indices = numpy.random.choice(X_test_seq.shape[0], no_plots, replace=False)
    counter = 0
    for idx in indices:
        counter += 1
        print("fraction done: ")
        print(counter / no_plots)

        coverage_predicted = model.predict([X_test_seq[idx:idx+1], X_test_annot[idx:idx+1]])[0]
        coverage_predicted = coverage_predicted.flatten()
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

        # Peaks
        observed_peaks, observed_properties = find_peaks(normalized_coverage, width=10, prominence=0.05)
        predicted_peaks, predicted_properties = find_peaks(coverage_predicted, width=10, prominence=0.05)
        # Plotting
        pyplot.style.use('ggplot')
        pyplot.figure(figsize=(12, 6))
        ymax = normalized_coverage.max() + (normalized_coverage.max() * 0.1)
        pyplot.plot(x_coord, normalized_coverage, color="blue", label='Observed Coverage Normalized by Gene Expression')
        pyplot.plot(observed_peaks, normalized_coverage[observed_peaks], "x", color='midnightblue', label='Observed Peaks')
        pyplot.plot(x_coord, coverage_predicted, color="purple", label='Predicted Coverage')
        pyplot.plot(predicted_peaks, coverage_predicted[predicted_peaks], "x", color='indigo', label='Predicted Peaks')
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
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    x_coord = numpy.arange(0, no_bin)               
    no_plots = int(no_plots)
    if random:
        numpy.random.seed(42) 
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


def plot_simulated_predictions(model, model_name, X_test_seq, X_test_annot, no_plots, no_bin, outpath, dataset_name, binsize, log_scale, anno_features, only_seq, print_GC):
    outpath = outpath + dataset_name+ "_outputs" + "/" + f"prediction_plots_{model_name}/"
    print("saving plots to: ", outpath)
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    x_coord = numpy.arange(0, no_bin)               
    no_plots = int(no_plots)
    #numpy.random.seed(42) 
    #indices = numpy.random.choice(X_test_seq.shape[0], no_plots, replace=False)
    indices = range(no_plots)
    print("indices: ", indices)
    counter = 0
    feature_indices = {feature: idx for idx, feature in enumerate(anno_features)}
    
    for feature_name in anno_features:
        feature_data = X_test_annot[0, :, feature_indices[feature_name]]

        if feature_name == 'gene_vector':
            gene_start = numpy.where(feature_data == 1)[0][0]
            gene_end = numpy.where(feature_data == 1)[0][1]
        if feature_name == 'promoter_vector':
            promoter = numpy.where(feature_data == 1)[0]
        if feature_name == 'terminator_vector':
            terminator_start = numpy.where(feature_data == 1)[0][0]
            terminator_end = numpy.where(feature_data == 1)[0][1]

    for idx in indices:
        counter += 1
        print("fraction done: ")
        print(counter / no_plots)
        print("idx: ", idx)

        input_1 = tf.cast(X_test_seq[idx:idx+1], tf.float32)
        input_2 = tf.cast(X_test_annot[idx:idx+1], tf.float32)

        gc_content = calculate_gc_content(input_1, window_size=40)
        coverage_predicted = model.predict([input_1, input_2])[0]
        coverage_predicted = coverage_predicted.flatten()

        predicted_peaks, predicted_properties = find_peaks(coverage_predicted, width=10, prominence=0.05)
        # Plotting
        pyplot.style.use('ggplot')
        pyplot.figure(figsize=(12, 6))
        ymax = coverage_predicted.max() + (coverage_predicted.max() * 0.1)
        pyplot.plot(x_coord, coverage_predicted, color="purple", label='Predicted Coverage From Simulated Data')
        pyplot.plot(predicted_peaks, coverage_predicted[predicted_peaks], "x", color='indigo', label='Predicted Peaks')
        pyplot.title(f"Predicted Coverage from Simulated Dataser {dataset_name}")
        pyplot.xlabel('Nucleotides')
        pyplot.ylabel('Normalized Coverage')
        if print_GC:
            gc_content_normalized = gc_content * ymax  
            pyplot.plot(x_coord, gc_content_normalized, 'r--', label='GC Content (%)')
        gene_y = -(0.05 * ymax)
        if log_scale == True:
            pyplot.ylabel('Log10(Normalized Coverage)')
            pyplot.yscale('log')
        pyplot.ylim(ymin=-0.2*ymax, ymax=ymax) 
        pyplot.xticks([0, round(no_bin/4), round(no_bin/2), round(3*no_bin/4), round(no_bin)], [0, round(no_bin/4), round(no_bin/2), round(3*no_bin/4), round(no_bin)])
        if 'gene_vector' in anno_features and not only_seq:
            pyplot.axvline(x=gene_start, ls="-", lw="1.2",color="green")
            pyplot.text(gene_start + (no_bin/80), ymax.max() * 0.9, 'gene start', verticalalignment='center', color='green')
            pyplot.axvline(x=gene_end, ls="-", lw="1.2",color="green")
            pyplot.text(gene_end + (no_bin/80), ymax.max() * 0.7, 'gene end', verticalalignment='center', color='green')
            pyplot.hlines(y=gene_y, xmin=gene_start/binsize, xmax=gene_end/binsize, colors='green', linestyles='solid')
            label_x_position = gene_start/binsize if gene_start >= 0 else gene_end/binsize
            pyplot.text(label_x_position, 1.5*gene_y, 'Gene body', color='green', fontsize=8)
        if 'promoter_vector' in anno_features and not only_seq:
            pyplot.axvline(x=promoter, ls="-.", lw="1.2",color="brown")
            pyplot.text(promoter + (no_bin/80), ymax.max() * 0.2, 'promoter', verticalalignment='center', color='brown')
        if 'terminator_vector' in anno_features and not only_seq:
            pyplot.hlines(y=2*gene_y, xmin=terminator_start/binsize, xmax=terminator_end/binsize, colors='red', linestyles='solid')
            label_x_position = terminator_start/binsize if terminator_start >= 0 else terminator_end/binsize
            pyplot.text(label_x_position, 2.5*gene_y, 'Terminator', color='red', fontsize=8)
        pyplot.legend()
        if log_scale == True:
            pyplot.savefig(outpath+model_name +str(idx)+ "predicted_coverage_log.png")
        else:
            pyplot.savefig(outpath+model_name +str(idx)+ "predicted_coverage.png")
        pyplot.close()
    return "Plots generated"


def calculate_gc_content(sequence, window_size=30):
    seq_length = sequence.shape[1]
    gc_content = numpy.zeros(seq_length)

    half_window = window_size // 2

    for i in range(seq_length):
        start = max(0, i - half_window)
        end = min(seq_length, i + half_window + 1)  # Adjust to include the end position in the window

        window = sequence[0,start:end,:]

        gc_count = numpy.sum(window[:, 1] + window[:, 2])  # Sum 'G' and 'C' counts
        window_bases = end - start  # Total number of bases in the current window
        gc_content[i] = (gc_count / window_bases)   # Calculate GC content percentage
    return gc_content


def plot_window_coverage_normalized_TU(normalized_coverage_with_windows_info, no_plots, outpath, dataset_name, promoter_df, terminator_df, gene_df, binsize,random = False):
    outpath = outpath + dataset_name+ "_outputs" + "/" + "window_coverage_plots/"
    if not os.path.exists(outpath):
        os.makedirs(outpath)
                   
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
        length_TU = abs(window_end - window_start)
        x_coord = numpy.arange(0, length_TU)
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

            if 0 <= gene_start < length_TU:
                print("start of: " + gene_row['Gene_Name']+ "is in this window")
                print("gene starts at:" + str(gene_row['Left']))
                print("gene ends at:" + str(gene_row['Right']))
                gene_starts.append(int(gene_start/binsize))
            if 0 <= gene_end < length_TU:
                print("end of: " + gene_row['Gene_Name'] + "is in this window")
                print("gene starts at:" + str(gene_row['Left']))
                print("gene ends at:" + str(gene_row['Right']))
                gene_ends.append(int(gene_end/binsize))
        
        for _, promoter_row in promoter_df.iterrows():
            
            promoter_position = int(promoter_row['Absolute_Plus_1_Position']) - window_start
            
            if 0 <= promoter_position < length_TU:
                promoters.append(int(promoter_position/binsize))

        for _, terminator in terminator_df.iterrows():
            terminator_start = int(terminator['Left_End_Position']) - window_start 
            terminator_end = int(terminator['Right_End_Position']) - window_start 

            # Ensure correct ordering of start and end
            terminator_start, terminator_end = min(terminator_start, terminator_end), max(terminator_start, terminator_end)

            if (0 <= terminator_start < length_TU):
                terminator_starts.append(int(terminator_start/binsize))
                
            if (0 <= terminator_end < length_TU):
                terminator_ends.append(int(terminator_end/binsize))

        # Plotting
        pyplot.style.use('ggplot')
        pyplot.figure(figsize=(12, 6))
        ymax = normalized_coverage.max() + (normalized_coverage.max() * 0.1)
        pyplot.plot(x_coord, normalized_coverage[:length_TU], color="blue", label='Observed Coverage Normalized by Gene Expression')
        pyplot.title(f"Normalized coverage over window: {window_start}-{window_end}")
        pyplot.xlabel('Bins')
        pyplot.ylabel('Normalized Coverage')
        pyplot.ylim(ymin=-0.2*ymax, ymax=ymax) 
        pyplot.xticks([0, round(length_TU/4), round(length_TU/2), round(3*length_TU/4), round(length_TU)], [0, round(length_TU/4), round(length_TU/2), round(3*length_TU/4), round(length_TU)])
        for gene_start in gene_starts:
            pyplot.axvline(x=gene_start, ls="-", lw="1.2",color="green")
            pyplot.text(gene_start + (length_TU/80), ymax.max() * 0.9, 'gene start', verticalalignment='center', color='green')
        for gene_end in gene_ends:
            pyplot.axvline(x=gene_end, ls="-", lw="1.2",color="green")
            pyplot.text(gene_end + (length_TU/80), ymax.max() * 0.7, 'gene end', verticalalignment='center', color='green')
        for promoter in promoters:
            pyplot.axvline(x=promoter, ls="-.", lw="1.2",color="brown")
            pyplot.text(promoter + (length_TU/80), ymax.max() * 0.2, 'promoter', verticalalignment='center', color='brown')
        
        # Plot gene bodies and terminators
        for _, gene_row in gene_df.iterrows():
            if gene_row['Left'] == 'None' or gene_row['Right']  == 'None':
                continue
            gene_start = int(gene_row['Left']) - window_start 
            gene_end = int(gene_row['Right']) - window_start 
            gene_start, gene_end = min(gene_start, gene_end), max(gene_start, gene_end)
            gene_name = gene_row['Gene_Name']

            # Check if gene start or end is within the window
            if (0 <= gene_start < length_TU) or (0 <= gene_end < length_TU):
                # Adjust for window boundaries
                gene_start = max(0, gene_start)
                gene_end = min(length_TU - 1, gene_end)
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
            if (0 <= terminator_start < length_TU) or (0 <= terminator_end < length_TU) or ((terminator_start < length_TU) and (terminator_end > length_TU)):
                if ((terminator_start < length_TU) and (terminator_end > length_TU)):
                    pyplot.hlines(y=2*gene_y, xmin=0, xmax=binsize, colors='red', linestyles='solid')
                    label_x_position = 0.5 * binsize
                    pyplot.text(label_x_position, 2.5*gene_y, terminator_name, color='red', fontsize=8)
                else:
                    terminator_start = max(0, terminator_start)
                    terminator_end = min(length_TU - 1, terminator_end)

                    pyplot.hlines(y=2*gene_y, xmin=terminator_start/binsize, xmax=terminator_end/binsize, colors='red', linestyles='solid')
                    label_x_position = terminator_start/binsize if terminator_start >= 0 else terminator_end/binsize
                    pyplot.text(label_x_position, 2.5*gene_y, terminator_name, color='red', fontsize=8)
        pyplot.legend()
        pyplot.savefig(outpath+str(window_start)+"_"+str(window_end)+"_window_coverage.png")
        print(outpath+str(window_start)+"_"+str(window_end)+"_window_coverage.png")
        pyplot.close()
    return "Plots generated"

def predictt(model, input_seq, input_anno):
    return model([input_seq, input_anno])

def predict_single(model, input_seq, input_anno):
    # Add a batch dimension: (1, seq_length, num_features)
    input_seq_batched = tf.expand_dims(input_seq, axis=0)
    input_anno_batched = tf.expand_dims(input_anno, axis=0)

    # Predict
    prediction = model([input_seq_batched, input_anno_batched])

    # Remove the batch dimension
    return tf.squeeze(prediction, axis=0)

def plot_predicted_vs_observed_TU(model, model_name, X_test_seq, X_test_annot, normalized_coverage_with_windows_info, no_plots, outpath, dataset_name, window_size, promoter_df, terminator_df, gene_df, binsize, log_scale):
    outpath = outpath + dataset_name+ "_outputs" + "/" + f"prediction_plots_{model_name}/"
    if not os.path.exists(outpath):
        os.makedirs(outpath)
                   
    no_plots = int(no_plots)
    
    total_entries = int(len(normalized_coverage_with_windows_info))
    indices = range(total_entries - no_plots, total_entries)
    counter = 0
    for idx in indices:
        counter += 1
        print(idx)
        print("fraction done: ")
        print(counter / no_plots)
        X_test_seq_single = tf.convert_to_tensor(X_test_seq[idx], dtype=tf.float32)  # Example for getting a single data point
        X_test_anno_single = tf.convert_to_tensor(X_test_annot[idx], dtype=tf.float32)

        coverage_predicted = predict_single(model, X_test_seq_single, X_test_anno_single)
        print("dimension of intput: ", coverage_predicted.shape)
        #coverage_predicted = model.predict([X_test_seq[idx:idx+1], X_test_annot[idx:idx+1]])[0]
        coverage_predicted = coverage_predicted.flatten()
        normalized_coverage = normalized_coverage_with_windows_info[idx, 2:]  # Skip the first two columns which contain the window start and end sites.
        window_start = int(normalized_coverage_with_windows_info[idx, 0])
        window_end = int(normalized_coverage_with_windows_info[idx, 1])
        length_TU = abs(window_end - window_start)
        x_coord = numpy.arange(0, length_TU)
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

            if 0 <= gene_start < length_TU:
                print("start of: " + gene_row['Gene_Name']+ "is in this window")
                print("gene starts at:" + str(gene_row['Left']))
                print("gene ends at:" + str(gene_row['Right']))
                gene_starts.append(int(gene_start/binsize))
            if 0 <= gene_end < length_TU:
                print("end of: " + gene_row['Gene_Name'] + "is in this window")
                print("gene starts at:" + str(gene_row['Left']))
                print("gene ends at:" + str(gene_row['Right']))
                gene_ends.append(int(gene_end/binsize))
        
        for _, promoter_row in promoter_df.iterrows():
            
            promoter_position = int(promoter_row['Absolute_Plus_1_Position']) - window_start
            
            if 0 <= promoter_position < length_TU:
                promoters.append(int(promoter_position/binsize))

        for _, terminator in terminator_df.iterrows():
            terminator_start = int(terminator['Left_End_Position']) - window_start 
            terminator_end = int(terminator['Right_End_Position']) - window_start 

            # Ensure correct ordering of start and end
            terminator_start, terminator_end = min(terminator_start, terminator_end), max(terminator_start, terminator_end)

            if (0 <= terminator_start < length_TU):
                terminator_starts.append(int(terminator_start/binsize))
                
            if (0 <= terminator_end < length_TU):
                terminator_ends.append(int(terminator_end/binsize))

        observed_peaks, observed_properties = find_peaks(normalized_coverage, width=10, prominence=0.05)
        predicted_peaks, predicted_properties = find_peaks(coverage_predicted, width=10, prominence=0.05)
        
        pyplot.plot(observed_peaks, normalized_coverage[observed_peaks], "x", color='midnightblue', label='Observed Peaks')
        pyplot.plot(predicted_peaks, coverage_predicted[predicted_peaks], "x", color='indigo', label='Predicted Peaks')
        pyplot.style.use('ggplot')
        pyplot.figure(figsize=(12, 6))
        ymax = normalized_coverage.max() + (normalized_coverage.max() * 0.1)
        pyplot.plot(x_coord, normalized_coverage[:length_TU], color="blue", label='Observed Coverage Normalized by Gene Expression')
        pyplot.plot(x_coord, coverage_predicted[:length_TU], color="purple", label='Predicted Coverage')
        pyplot.title(f"Normalized coverage over window: {window_start}-{window_end}")
        pyplot.xlabel('Bins')
        pyplot.ylabel('Normalized Coverage')
        pyplot.ylim(ymin=-0.2*ymax, ymax=ymax) 
        pyplot.xticks([0, round(length_TU/4), round(length_TU/2), round(3*length_TU/4), round(length_TU)], [0, round(length_TU/4), round(length_TU/2), round(3*length_TU/4), round(length_TU)])
        for gene_start in gene_starts:
            pyplot.axvline(x=gene_start, ls="-", lw="1.2",color="green")
            pyplot.text(gene_start + (length_TU/80), ymax.max() * 0.9, 'gene start', verticalalignment='center', color='green')
        for gene_end in gene_ends:
            pyplot.axvline(x=gene_end, ls="-", lw="1.2",color="green")
            pyplot.text(gene_end + (length_TU/80), ymax.max() * 0.7, 'gene end', verticalalignment='center', color='green')
        for promoter in promoters:
            pyplot.axvline(x=promoter, ls="-.", lw="1.2",color="brown")
            pyplot.text(promoter + (length_TU/80), ymax.max() * 0.2, 'promoter', verticalalignment='center', color='brown')
        
        # Plot gene bodies and terminators
        for _, gene_row in gene_df.iterrows():
            if gene_row['Left'] == 'None' or gene_row['Right']  == 'None':
                continue
            gene_start = int(gene_row['Left']) - window_start 
            gene_end = int(gene_row['Right']) - window_start 
            gene_start, gene_end = min(gene_start, gene_end), max(gene_start, gene_end)
            gene_name = gene_row['Gene_Name']

            # Check if gene start or end is within the window
            if (0 <= gene_start < length_TU) or (0 <= gene_end < length_TU):
                # Adjust for window boundaries
                gene_start = max(0, gene_start)
                gene_end = min(length_TU - 1, gene_end)
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
            if (0 <= terminator_start < length_TU) or (0 <= terminator_end < length_TU) or ((terminator_start < length_TU) and (terminator_end > length_TU)):
                if ((terminator_start < length_TU) and (terminator_end > length_TU)):
                    pyplot.hlines(y=2*gene_y, xmin=0, xmax=binsize, colors='red', linestyles='solid')
                    label_x_position = 0.5 * binsize
                    pyplot.text(label_x_position, 2.5*gene_y, terminator_name, color='red', fontsize=8)
                else:
                    terminator_start = max(0, terminator_start)
                    terminator_end = min(length_TU - 1, terminator_end)

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


def plot_predicted_vs_observed_TU_during_training(model_name, Y_pred, Y_test, outpath, dataset_name, promoter_df, terminator_df, gene_df, binsize, window_start, window_end, log_scale, pad_symbol):
    outpath = outpath + dataset_name+ "_outputs" + "/" + f"prediction_plots_{model_name}/"
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    normalized_coverage = Y_test
    coverage_predicted = Y_pred
    length_TU = int(abs(window_end - window_start))
    print("within plotting function")
    print("length_TU: ", length_TU)
    
    x_coord = numpy.arange(0, length_TU)
    print("initial length x_coord: ", len(x_coord))
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

        if 0 <= gene_start < length_TU:
            #print("start of: " + gene_row['Gene_Name']+ "is in this window")
            #print("gene starts at:" + str(gene_row['Left']))
            #print("gene ends at:" + str(gene_row['Right']))
            gene_starts.append(int(gene_start/binsize))
        if 0 <= gene_end < length_TU:
            #print("end of: " + gene_row['Gene_Name'] + "is in this window")
            #print("gene starts at:" + str(gene_row['Left']))
            #print("gene ends at:" + str(gene_row['Right']))
            gene_ends.append(int(gene_end/binsize))
    
    for _, promoter_row in promoter_df.iterrows():
        
        promoter_position = int(promoter_row['Absolute_Plus_1_Position']) - window_start
        
        if 0 <= promoter_position < length_TU:
            promoters.append(int(promoter_position/binsize))

    for _, terminator in terminator_df.iterrows():
        terminator_start = int(terminator['Left_End_Position']) - window_start 
        terminator_end = int(terminator['Right_End_Position']) - window_start 

        # Ensure correct ordering of start and end
        terminator_start, terminator_end = min(terminator_start, terminator_end), max(terminator_start, terminator_end)

        if (0 <= terminator_start < length_TU):
            terminator_starts.append(int(terminator_start/binsize))
            
        if (0 <= terminator_end < length_TU):
            terminator_ends.append(int(terminator_end/binsize))
    
    coverage_predicted = coverage_predicted.flatten()
    normalized_coverage = normalized_coverage.flatten()
    normalized_coverage = numpy.round(normalized_coverage, 2)
    pad_symbol_rounded = numpy.round(pad_symbol, 2)

    # Find the first index where there are at least 3 subsequent pad symbols
    count = 0
    first_pad_index = None
    for i in range(len(normalized_coverage)):
        if normalized_coverage[i] == pad_symbol_rounded:
            count += 1
            if count >= 10:
                first_pad_index = i - 10  # Adjust for 3 subsequent pad symbols
                break
        else:
            count = 0

    # If at least 3 subsequent pad symbols were found, truncate observed and predicted arrays
    if first_pad_index is not None:
        #print("pad symbol found")
        #print("first_pad_index: ", first_pad_index)
        normalized_coverage = normalized_coverage[:first_pad_index]
        coverage_predicted = coverage_predicted[:first_pad_index]
        x_coord = x_coord[:first_pad_index]
        #print("length x_coord: ", len(x_coord))
        #print("length normalized_coverage: ", len(normalized_coverage))
        #print("length coverage_predicted: ", len(coverage_predicted))

    else:
        #print("no pad symbol found")
        first_pad_index = len(normalized_coverage)
        normalized_coverage = normalized_coverage[:first_pad_index]
        coverage_predicted = coverage_predicted[:first_pad_index]
        x_coord = x_coord[:first_pad_index]
        #print("length x_coord: ", len(x_coord))
        #print("length normalized_coverage: ", len(normalized_coverage))
        #print("length coverage_predicted: ", len(coverage_predicted))
    observed_peaks, observed_properties = find_peaks(normalized_coverage, width=10, prominence=0.05)
    predicted_peaks, predicted_properties = find_peaks(coverage_predicted, width=10, prominence=0.05)
   
    
    pyplot.style.use('ggplot')
    pyplot.figure(figsize=(12, 6))
    # calculate if predicted coverage is higher than observed coverage
    pred_max = coverage_predicted.max()
    obs_max = normalized_coverage.max()
    if pred_max >= obs_max:
        ymax = pred_max + (pred_max * 0.1)
    else:
        ymax = obs_max + (obs_max * 0.1)
    pyplot.plot(x_coord, normalized_coverage[:length_TU], color="blue", label='Observed Coverage Normalized by Gene Expression')
    pyplot.plot(x_coord, coverage_predicted[:length_TU], color="purple", label='Predicted Coverage')
    pyplot.plot(observed_peaks, normalized_coverage[observed_peaks], "x", color='midnightblue', label='Observed Peaks')
    pyplot.plot(predicted_peaks, coverage_predicted[predicted_peaks], "x", color='indigo', label='Predicted Peaks')
    pyplot.title(f"Normalized coverage over window: {window_start}-{window_end}")
    pyplot.xlabel('Bins')
    pyplot.ylabel('Normalized Coverage')
    pyplot.ylim(ymin=-0.2*ymax, ymax=ymax) 
    pyplot.xticks([0, round(length_TU/4), round(length_TU/2), round(3*length_TU/4), round(length_TU)], [0, round(length_TU/4), round(length_TU/2), round(3*length_TU/4), round(length_TU)])
    for gene_start in gene_starts:
        pyplot.axvline(x=gene_start, ls="-", lw="1.2",color="green")
        pyplot.text(gene_start + (length_TU/80), ymax.max() * 0.9, 'gene start', verticalalignment='center', color='green')
    for gene_end in gene_ends:
        pyplot.axvline(x=gene_end, ls="-", lw="1.2",color="green")
        pyplot.text(gene_end + (length_TU/80), ymax.max() * 0.7, 'gene end', verticalalignment='center', color='green')
    for promoter in promoters:
        pyplot.axvline(x=promoter, ls="-.", lw="1.2",color="brown")
        pyplot.text(promoter + (length_TU/80), ymax.max() * 0.2, 'promoter', verticalalignment='center', color='brown')
    
    # Plot gene bodies and terminators
    for _, gene_row in gene_df.iterrows():
        if gene_row['Left'] == 'None' or gene_row['Right']  == 'None':
            continue
        gene_start = int(gene_row['Left']) - window_start 
        gene_end = int(gene_row['Right']) - window_start 
        gene_start, gene_end = min(gene_start, gene_end), max(gene_start, gene_end)
        gene_name = gene_row['Gene_Name']

        # Check if gene start or end is within the window
        if (0 <= gene_start < length_TU) or (0 <= gene_end < length_TU):
            # Adjust for window boundaries
            gene_start = max(0, gene_start)
            gene_end = min(length_TU - 1, gene_end)
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
        if (0 <= terminator_start < length_TU) or (0 <= terminator_end < length_TU) or ((terminator_start < length_TU) and (terminator_end > length_TU)):
            if ((terminator_start < length_TU) and (terminator_end > length_TU)):
                pyplot.hlines(y=2*gene_y, xmin=0, xmax=binsize, colors='red', linestyles='solid')
                label_x_position = 0.5 * binsize
                pyplot.text(label_x_position, 2.5*gene_y, terminator_name, color='red', fontsize=8)
            else:
                terminator_start = max(0, terminator_start)
                terminator_end = min(length_TU - 1, terminator_end)

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

def plot_predicted_vs_observed_TU_during_training_probab(model_name, Y_pred, Y_test, outpath, dataset_name, promoter_df, terminator_df, gene_df, binsize, window_start, window_end, log_scale, pad_symbol):
    outpath = outpath + dataset_name+ "_outputs" + "/" + f"prediction_plots_{model_name}/"
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    normalized_coverage = Y_test
    coverage_predicted = Y_pred
    length_TU = int(abs(window_end - window_start))
    print("within plotting function")
    print("length_TU: ", length_TU)
    
    x_coord = numpy.arange(0, length_TU)
    print("initial length x_coord: ", len(x_coord))
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

        if 0 <= gene_start < length_TU:
            #print("start of: " + gene_row['Gene_Name']+ "is in this window")
            #print("gene starts at:" + str(gene_row['Left']))
            #print("gene ends at:" + str(gene_row['Right']))
            gene_starts.append(int(gene_start/binsize))
        if 0 <= gene_end < length_TU:
            #print("end of: " + gene_row['Gene_Name'] + "is in this window")
            #print("gene starts at:" + str(gene_row['Left']))
            #print("gene ends at:" + str(gene_row['Right']))
            gene_ends.append(int(gene_end/binsize))
    
    for _, promoter_row in promoter_df.iterrows():
        
        promoter_position = int(promoter_row['Absolute_Plus_1_Position']) - window_start
        
        if 0 <= promoter_position < length_TU:
            promoters.append(int(promoter_position/binsize))

    for _, terminator in terminator_df.iterrows():
        terminator_start = int(terminator['Left_End_Position']) - window_start 
        terminator_end = int(terminator['Right_End_Position']) - window_start 

        # Ensure correct ordering of start and end
        terminator_start, terminator_end = min(terminator_start, terminator_end), max(terminator_start, terminator_end)

        if (0 <= terminator_start < length_TU):
            terminator_starts.append(int(terminator_start/binsize))
            
        if (0 <= terminator_end < length_TU):
            terminator_ends.append(int(terminator_end/binsize))
    
    coverage_predicted = coverage_predicted.flatten()
    normalized_coverage = normalized_coverage.flatten()
    normalized_coverage = numpy.round(normalized_coverage, 4)
    pad_symbol_rounded = numpy.round(pad_symbol, 4)

    # Find the first index where there are at least 3 subsequent pad symbols
    count = 0
    first_pad_index = None
    for i in range(len(normalized_coverage)):
        if normalized_coverage[i] == pad_symbol_rounded:
            count += 1
            if count >= 10:
                first_pad_index = i - 10  # Adjust for 3 subsequent pad symbols
                break
        else:
            count = 0

    # If at least 3 subsequent pad symbols were found, truncate observed and predicted arrays
    if first_pad_index is not None:
        #print("pad symbol found")
        #print("first_pad_index: ", first_pad_index)
        normalized_coverage = normalized_coverage[:first_pad_index]
        coverage_predicted = coverage_predicted[:first_pad_index]
        x_coord = x_coord[:first_pad_index]
        #print("length x_coord: ", len(x_coord))
        #print("length normalized_coverage: ", len(normalized_coverage))
        #print("length coverage_predicted: ", len(coverage_predicted))

    else:
        #print("no pad symbol found")
        first_pad_index = len(normalized_coverage)
        normalized_coverage = normalized_coverage[:first_pad_index]
        coverage_predicted = coverage_predicted[:first_pad_index]
        x_coord = x_coord[:first_pad_index]
        #print("length x_coord: ", len(x_coord))
        #print("length normalized_coverage: ", len(normalized_coverage))
        #print("length coverage_predicted: ", len(coverage_predicted))
    observed_peaks, observed_properties = find_peaks(normalized_coverage, width=10, prominence=0.00625)
    predicted_peaks, predicted_properties = find_peaks(coverage_predicted, width=10, prominence=0.00625)
   
    
    pyplot.style.use('ggplot')
    pyplot.figure(figsize=(12, 6))
    # calculate if predicted coverage is higher than observed coverage
    pred_max = coverage_predicted.max()
    obs_max = normalized_coverage.max()
    if pred_max >= obs_max:
        ymax = pred_max + (pred_max * 0.1)
    else:
        ymax = obs_max + (obs_max * 0.1)
    pyplot.plot(x_coord, normalized_coverage[:length_TU], color="blue", label='Observed Coverage Normalized by Gene Expression')
    pyplot.plot(x_coord, coverage_predicted[:length_TU], color="purple", label='Predicted Coverage')
    pyplot.plot(observed_peaks, normalized_coverage[observed_peaks], "x", color='midnightblue', label='Observed Peaks')
    pyplot.plot(predicted_peaks, coverage_predicted[predicted_peaks], "x", color='indigo', label='Predicted Peaks')
    pyplot.title(f"Normalized coverage over window: {window_start}-{window_end}")
    pyplot.xlabel('Bins')
    pyplot.ylabel('Normalized Coverage')
    pyplot.ylim(ymin=-0.2*ymax, ymax=ymax) 
    pyplot.xticks([0, round(length_TU/4), round(length_TU/2), round(3*length_TU/4), round(length_TU)], [0, round(length_TU/4), round(length_TU/2), round(3*length_TU/4), round(length_TU)])
    for gene_start in gene_starts:
        pyplot.axvline(x=gene_start, ls="-", lw="1.2",color="green")
        pyplot.text(gene_start + (length_TU/80), ymax.max() * 0.9, 'gene start', verticalalignment='center', color='green')
    for gene_end in gene_ends:
        pyplot.axvline(x=gene_end, ls="-", lw="1.2",color="green")
        pyplot.text(gene_end + (length_TU/80), ymax.max() * 0.7, 'gene end', verticalalignment='center', color='green')
    for promoter in promoters:
        pyplot.axvline(x=promoter, ls="-.", lw="1.2",color="brown")
        pyplot.text(promoter + (length_TU/80), ymax.max() * 0.2, 'promoter', verticalalignment='center', color='brown')
    
    # Plot gene bodies and terminators
    for _, gene_row in gene_df.iterrows():
        if gene_row['Left'] == 'None' or gene_row['Right']  == 'None':
            continue
        gene_start = int(gene_row['Left']) - window_start 
        gene_end = int(gene_row['Right']) - window_start 
        gene_start, gene_end = min(gene_start, gene_end), max(gene_start, gene_end)
        gene_name = gene_row['Gene_Name']

        # Check if gene start or end is within the window
        if (0 <= gene_start < length_TU) or (0 <= gene_end < length_TU):
            # Adjust for window boundaries
            gene_start = max(0, gene_start)
            gene_end = min(length_TU - 1, gene_end)
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
        if (0 <= terminator_start < length_TU) or (0 <= terminator_end < length_TU) or ((terminator_start < length_TU) and (terminator_end > length_TU)):
            if ((terminator_start < length_TU) and (terminator_end > length_TU)):
                pyplot.hlines(y=2*gene_y, xmin=0, xmax=binsize, colors='red', linestyles='solid')
                label_x_position = 0.5 * binsize
                pyplot.text(label_x_position, 2.5*gene_y, terminator_name, color='red', fontsize=8)
            else:
                terminator_start = max(0, terminator_start)
                terminator_end = min(length_TU - 1, terminator_end)

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