import numpy as np
from tensorflow.keras.callbacks import Callback
from matplotlib import pyplot

class NaNChecker(Callback):
    def on_batch_end(self, batch, logs=None):
        if np.isnan(logs['loss']):
            print('Batch %d: Invalid loss, terminating training' % (batch))
            self.model.stop_training = True


def plot_predicted_vs_observed(X_test, Y_test, model, no_plots, no_bin, outpath, model_name):
    
    x_coord = np.arange(0, no_bin)
    indices = np.random.choice(X_test.shape[0], no_plots, replace=False)

    for idx in indices:
        
        # Extracting the observed and predicted coverage
        coverage_observed = Y_test[idx]
        coverage_predicted = model.predict(X_test[idx:idx+1])[0]
        binary_predictions = (coverage_predicted > 0.5).astype(int)

        # Plotting
        pyplot.style.use('ggplot')
        pyplot.figure(figsize=(12, 6))
        pyplot.plot(x_coord, coverage_observed, color="blue", label='Observed binary (peak yes/no)')
        #pyplot.plot(x_coord, binary_predictions, linestyle='--', color="red", label='Predicted binary (P(peak)>0.5)')
        pyplot.plot(x_coord, coverage_predicted, color="purple", label='Predicted P(peak)')
        pyplot.title(f"Predicted vs. observed coverage over example window {idx}")
        pyplot.xlabel('Bins')
        pyplot.ylabel('Normalized coverage summed over bam files')
        ymax = max(coverage_observed.max(), binary_predictions.max()) + (max(coverage_observed.max(), binary_predictions.max()) * 0.1)
        pyplot.ylim(ymin=0, ymax=ymax)
        pyplot.xticks([0, round(no_bin/4), round(no_bin/2), round(3*no_bin/4), round(no_bin)], [0, round(no_bin/4), round(no_bin/2), round(3*no_bin/4), round(no_bin)])
        pyplot.legend()
        pyplot.savefig(outpath + "_" + str(idx) + str(model_name) + "_obs_vs_pred.png")
        pyplot.close()

    return("Plots generated")


def plot_predicted_vs_observed_window_info(X_test, Y_test, model, no_plots, no_bin, outpath, model_name, window_size, operon_df, gene_df, binsize):
    
    x_coord = np.arange(0, no_bin)
    indices = np.random.choice(X_test.shape[0], no_plots, replace=False)

    for idx in indices:
        # Extracting the observed and predicted coverage, skipping the first two columns in Y_test
        coverage_observed = Y_test[idx, 2:]  # Skip the first two columns
        coverage_predicted = model.predict(X_test[idx:idx+1])[0]
        binary_predictions = (coverage_predicted > 0.5).astype(int)
        window_start = int(Y_test[idx, 0])
        window_end = int(Y_test[idx, 1])

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
                #print("end of: " + operon['operonName'] + "is in this window")
                #print("operon starts at:" + str(operon['firstGeneLeftPos']))
                #print("operon ends at:" + str(operon['lastGeneRightPos']))
        # Plotting
        pyplot.style.use('ggplot')
        pyplot.figure(figsize=(12, 6))
        ymax = max(coverage_observed.max(), binary_predictions.max()) + (max(coverage_observed.max(), binary_predictions.max()) * 0.1)
        pyplot.plot(x_coord, coverage_observed, color="blue", label='Observed binary (peak yes/no)')
        #pyplot.plot(x_coord, binary_predictions, linestyle='--', color="red", label='Predicted binary (P(peak)>0.5)')
        pyplot.plot(x_coord, coverage_predicted, color="purple", label='Predicted P(peak)')
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
        pyplot.title(f"Predicted vs. observed coverage over example window {idx}")
        pyplot.xlabel('Bins')
        pyplot.ylabel('Normalized coverage summed over bam files')
        pyplot.ylim(ymin=0, ymax=ymax)
        pyplot.xticks([0, round(no_bin/4), round(no_bin/2), round(3*no_bin/4), round(no_bin)], [0, round(no_bin/4), round(no_bin/2), round(3*no_bin/4), round(no_bin)])
        pyplot.legend()
        pyplot.savefig(outpath + "_" + str(idx) + "_" + str(model_name) + "_obs_vs_pred.png")
        pyplot.close()

    return "Plots generated"


def plot_predicted_vs_observed_window_info_lines(X_test, Y_test, model, no_plots, no_bin, outpath, model_name, window_size, operon_df, gene_df, binsize):
    x_coord = np.arange(0, no_bin)               
    indices = np.random.choice(X_test.shape[0], no_plots, replace=False)

    for idx in indices:
        coverage_observed = Y_test[idx, 2:]  # Skip the first two columns
        coverage_predicted = model.predict(X_test[idx:idx+1])[0]
        binary_predictions = (coverage_predicted > 0.5).astype(int)
        window_start = int(Y_test[idx, 0])
        window_end = int(Y_test[idx, 1])
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
        ymax = max(coverage_observed.max(), binary_predictions.max()) + (max(coverage_observed.max(), binary_predictions.max()) * 0.1)
        pyplot.plot(x_coord, coverage_observed, color="blue", label='Observed Coverage Binary (peak yes/no)')
        pyplot.plot(x_coord, coverage_predicted, color="purple", label='Predicted Peak Probability')
        pyplot.title(f"Predicted vs. Observed Coverage over Window {window_start}-{window_end}")
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
        pyplot.savefig(outpath + "_" + str(idx) + "_" + str(model_name) + "_obs_vs_pred.png")
        pyplot.close()
    return "Plots generated"


def plot_coverage_predicted_vs_observed_window_info_lines_log(X_test, Y_test, model, no_plots, no_bin, outpath, model_name, window_size, operon_df, gene_df, binsize):
    x_coord = np.arange(0, no_bin)               
    indices = np.random.choice(X_test.shape[0], no_plots, replace=False)

    for idx in indices:
        small_constant = 1e-8
        
        coverage_observed = Y_test[idx, 2:]  # Skip the first two columns
        #coverage_observed = np.log10(coverage_observed + small_constant)
        coverage_predicted = model.predict(X_test[idx:idx+1])[0]
        #coverage_predicted = 10 ** coverage_predicted
        window_start = int(Y_test[idx, 0])
        window_end = int(Y_test[idx, 1])
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
        ymax = max(coverage_observed.max(), coverage_predicted.max()) + (max(coverage_observed.max(), coverage_predicted.max()) * 0.1)
        ymin = min(coverage_observed.min(), coverage_predicted.min()) + (min(coverage_observed.min(), coverage_predicted.min()) * 0.1)
        span = np.absolute(ymax-ymin)
        pyplot.plot(x_coord, coverage_observed, color="blue", label='Observed Coverage')
        pyplot.plot(x_coord, coverage_predicted, color="purple", label='Predicted Coverage')
        pyplot.title(f"Predicted vs. Observed Coverage over Window {window_start}-{window_end}")
        pyplot.xlabel('Bins')
        pyplot.ylabel('Coverage')
        #pyplot.yscale('log')
        #pyplot.ylim(ymin=max(1e-3, min(coverage_predicted[coverage_predicted > 0].min(),coverage_observed[coverage_observed > 0].min())), ymax=ymax)
        pyplot.ylim(ymin=ymin - 0.1 * span, ymax=ymax) # -0.2
        pyplot.xticks([0, round(no_bin/4), round(no_bin/2), round(3*no_bin/4), round(no_bin)], [0, round(no_bin/4), round(no_bin/2), round(3*no_bin/4), round(no_bin)])
        #for gene_start in gene_starts:
        #    pyplot.axvline(x=gene_start, ls="-", lw="1.2",color="green")
        #    pyplot.text(gene_start + (no_bin/80), ymin + span * 0.9, 'gene start', verticalalignment='center', color='green')
        #for gene_end in gene_ends:
        #    pyplot.axvline(x=gene_end, ls="-", lw="1.2",color="green")
        #    pyplot.text(gene_end + (no_bin/80), ymin + span * 0.3, 'gene end', verticalalignment='center', color='green')
        #for operon_start in operon_starts:
        #    pyplot.axvline(x=operon_start, ls="-.", lw="1",color="red")
        #    pyplot.text(operon_start + (no_bin/80), ymin + span * 0.8, 'operon start', verticalalignment='center', color='red')
        #for operon_end in operon_ends:
        #    pyplot.axvline(x=operon_end, ls="-.", lw="1",color="red")
        #    pyplot.text(operon_end + (no_bin/80), ymin + span * 0.2, 'operon end', verticalalignment='center', color='red')
        # Plot gene and operon bodies
        for _, gene_row in gene_df.iterrows():
            if gene_row['leftEndPos'] == 'None' or gene_row['rightEndPos']  == 'None':
                continue
            gene_start = int(gene_row['leftEndPos']) - window_start 
            gene_end = int(gene_row['rightEndPos']) - window_start 
            gene_start, gene_end = min(gene_start, gene_end), max(gene_start, gene_end)
            gene_name = gene_row['geneName']

            # Check if gene start or end is within the window
            if (0 <= gene_start < window_size) or (0 <= gene_end < window_size):
                # Adjust for window boundaries
                gene_start = max(0, gene_start)
                gene_end = min(window_size - 1, gene_end)
                gene_y = ymin - 0.01 * span
                pyplot.hlines(y=gene_y, xmin=gene_start/binsize, xmax=gene_end/binsize, colors='green', linestyles='solid')
                label_x_position = gene_start/binsize if gene_start >= 0 else gene_end/binsize
                pyplot.text(label_x_position, gene_y- 0.02 * span, gene_name, color='green', fontsize=8)

        for _, operon_row in operon_df.iterrows():
            operon_start = int(operon_row['firstGeneLeftPos']) 
            operon_end = int(operon_row['lastGeneRightPos']) 
            operon_start, operon_end = min(operon_start, operon_end), max(operon_start, operon_end)
            operon_name = operon_row['operonName']

            # Check if operon start or end is within the window
            if (window_start <= operon_start < window_end) or (window_start <= operon_end < window_end) or ((operon_start < window_start) and (operon_end > window_end)):
                if ((operon_start < window_start) and (operon_end > window_end)):
                    pyplot.hlines(y=gene_y- 0.03 * span, xmin=0, xmax=binsize, colors='red', linestyles='solid')
                    label_x_position = 0.5 * binsize
                    pyplot.text(label_x_position, gene_y- 0.05 * span, operon_name, color='red', fontsize=8)
                
                elif (window_start <= operon_start < window_end) and (window_start <= operon_end < window_end):
                    pyplot.hlines(y=gene_y- 0.03 * span, xmin=int((operon_start-window_start)/binsize), xmax=int((operon_end-window_start)/binsize), colors='red', linestyles='solid')
                    label_x_position = int((operon_start-window_start)/binsize) + 0.5 * (int((operon_end-window_start)/binsize)-int((operon_start-window_start)/binsize))
                    pyplot.text(label_x_position, gene_y- 0.05 * span, operon_name, color='red', fontsize=8)

                elif (operon_start < window_start) and (window_start <= operon_end < window_end):
                    pyplot.hlines(y=gene_y- 0.03 * span, xmin=0, xmax=int((operon_end-window_start)/binsize), colors='red', linestyles='solid')
                    label_x_position = 0.5 * int((operon_end-window_start)/binsize)
                    pyplot.text(label_x_position, gene_y- 0.05 * span, operon_name, color='red', fontsize=8)

                elif (window_start <= operon_start < window_end) and ( operon_end > window_end):
                    pyplot.hlines(y=gene_y- 0.03 * span, xmin=int((operon_start-window_start)/binsize), xmax=no_bin, colors='red', linestyles='solid')
                    label_x_position = int((operon_start-window_start)/binsize) + 0.5 * (no_bin-int((operon_start-window_start)/binsize))
                    pyplot.text(label_x_position, gene_y- 0.05 * span, operon_name, color='red', fontsize=8)

                    
        pyplot.legend()
        pyplot.savefig(outpath + "_" + str(idx)+ "_" + str(model_name)+  "_obs_vs_pred.png")
        pyplot.close()
    return "Plots generated"


def plot_coverage_predicted_vs_observed_window_info_lines_binary(X_test, Y_test, model, no_plots, no_bin, outpath, model_name, window_size, operon_df, gene_df, binsize):
    x_coord = np.arange(0, no_bin)               
    indices = np.random.choice(X_test.shape[0], no_plots, replace=False)

    for idx in indices:
        small_constant = 1e-8
        
        coverage_observed = Y_test[idx, 2:]  # Skip the first two columns
        #coverage_observed = np.log10(coverage_observed + small_constant)
        coverage_predicted = model.predict(X_test[idx:idx+1])[0]
        #coverage_predicted = 10 ** coverage_predicted
        window_start = int(Y_test[idx, 0])
        window_end = int(Y_test[idx, 1])
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
        ymax = 1.1
        ymin = -0.1
        span = np.absolute(ymax-ymin)
        pyplot.plot(x_coord, coverage_observed, color="blue", label='Observed Peak Binary (peak yes/no)')
        pyplot.plot(x_coord, coverage_predicted, color="purple", label='Predicted preak probability')
        pyplot.title(f"Predicted vs. Observed Coverage over Window {window_start}-{window_end}")
        pyplot.xlabel('Bins')
        pyplot.ylabel('Spacer Coverage Peaks')
        #pyplot.yscale('log')
        #pyplot.ylim(ymin=max(1e-3, min(coverage_predicted[coverage_predicted > 0].min(),coverage_observed[coverage_observed > 0].min())), ymax=ymax)
        pyplot.ylim(ymin=ymin - 0.1 * span, ymax=ymax) # -0.2
        pyplot.xticks([0, round(no_bin/4), round(no_bin/2), round(3*no_bin/4), round(no_bin)], [0, round(no_bin/4), round(no_bin/2), round(3*no_bin/4), round(no_bin)])
        #for gene_start in gene_starts:
        #    pyplot.axvline(x=gene_start, ls="-", lw="1.2",color="green")
        #    pyplot.text(gene_start + (no_bin/80), ymin + span * 0.9, 'gene start', verticalalignment='center', color='green')
        #for gene_end in gene_ends:
        #    pyplot.axvline(x=gene_end, ls="-", lw="1.2",color="green")
        #    pyplot.text(gene_end + (no_bin/80), ymin + span * 0.3, 'gene end', verticalalignment='center', color='green')
        #for operon_start in operon_starts:
        #    pyplot.axvline(x=operon_start, ls="-.", lw="1",color="red")
        #    pyplot.text(operon_start + (no_bin/80), ymin + span * 0.8, 'operon start', verticalalignment='center', color='red')
        #for operon_end in operon_ends:
        #    pyplot.axvline(x=operon_end, ls="-.", lw="1",color="red")
        #    pyplot.text(operon_end + (no_bin/80), ymin + span * 0.2, 'operon end', verticalalignment='center', color='red')
        # Plot gene and operon bodies
        for _, gene_row in gene_df.iterrows():
            if gene_row['leftEndPos'] == 'None' or gene_row['rightEndPos']  == 'None':
                continue
            gene_start = int(gene_row['leftEndPos']) - window_start 
            gene_end = int(gene_row['rightEndPos']) - window_start 
            gene_start, gene_end = min(gene_start, gene_end), max(gene_start, gene_end)
            gene_name = gene_row['geneName']

            # Check if gene start or end is within the window
            if (0 <= gene_start < window_size) or (0 <= gene_end < window_size):
                # Adjust for window boundaries
                gene_start = max(0, gene_start)
                gene_end = min(window_size - 1, gene_end)
                gene_y = ymin - 0.01 * span
                pyplot.hlines(y=gene_y, xmin=gene_start/binsize, xmax=gene_end/binsize, colors='green', linestyles='solid')
                label_x_position = gene_start/binsize if gene_start >= 0 else gene_end/binsize
                pyplot.text(label_x_position, gene_y- 0.02 * span, gene_name, color='green', fontsize=8)

        for _, operon_row in operon_df.iterrows():
            operon_start = int(operon_row['firstGeneLeftPos']) 
            operon_end = int(operon_row['lastGeneRightPos']) 
            operon_start, operon_end = min(operon_start, operon_end), max(operon_start, operon_end)
            operon_name = operon_row['operonName']

            # Check if operon start or end is within the window
            if (window_start <= operon_start < window_end) or (window_start <= operon_end < window_end) or ((operon_start < window_start) and (operon_end > window_end)):
                if ((operon_start < window_start) and (operon_end > window_end)):
                    pyplot.hlines(y=gene_y- 0.03 * span, xmin=0, xmax=binsize, colors='red', linestyles='solid')
                    label_x_position = 0.5 * binsize
                    pyplot.text(label_x_position, gene_y- 0.05 * span, operon_name, color='red', fontsize=8)
                
                elif (window_start <= operon_start < window_end) and (window_start <= operon_end < window_end):
                    pyplot.hlines(y=gene_y- 0.03 * span, xmin=int((operon_start-window_start)/binsize), xmax=int((operon_end-window_start)/binsize), colors='red', linestyles='solid')
                    label_x_position = int((operon_start-window_start)/binsize) + 0.5 * (int((operon_end-window_start)/binsize)-int((operon_start-window_start)/binsize))
                    pyplot.text(label_x_position, gene_y- 0.05 * span, operon_name, color='red', fontsize=8)

                elif (operon_start < window_start) and (window_start <= operon_end < window_end):
                    pyplot.hlines(y=gene_y- 0.03 * span, xmin=0, xmax=int((operon_end-window_start)/binsize), colors='red', linestyles='solid')
                    label_x_position = 0.5 * int((operon_end-window_start)/binsize)
                    pyplot.text(label_x_position, gene_y- 0.05 * span, operon_name, color='red', fontsize=8)

                elif (window_start <= operon_start < window_end) and ( operon_end > window_end):
                    pyplot.hlines(y=gene_y- 0.03 * span, xmin=int((operon_start-window_start)/binsize), xmax=no_bin, colors='red', linestyles='solid')
                    label_x_position = int((operon_start-window_start)/binsize) + 0.5 * (no_bin-int((operon_start-window_start)/binsize))
                    pyplot.text(label_x_position, gene_y- 0.05 * span, operon_name, color='red', fontsize=8)

                    
        pyplot.legend()
        pyplot.savefig(outpath + "_" + str(idx)+ "_" + str(model_name)+  "_obs_vs_pred.png")
        pyplot.close()
    return "Plots generated"