from sklearn.utils import shuffle
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

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

def normalize_coverage_per_gene(coverage_data, sequence_data, gene_spacer_counts_df, no_bin, binsize): 
    normalized_coverage_data = coverage_data.copy()
    windows_to_delete = [] # If there is some window that isn't covered by any gene, it is deleted.
    counter = 0
    nrow = coverage_data.shape[0]
    for index, row in enumerate(coverage_data):
        counter += 1
        #print("perc. done: " + str(100*(counter/nrow)))
        window_start = int(row[0]) 
        window_end = int(row[1]) 
        #print("WINDOW: " + str(window_start)+" to " + str(window_end))
        normalization_factors = np.ones(no_bin)

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
        if status == "only_ones":
            #print("DANGER DANGER DANGER ONLY ONEES ALAALALALALLAA")
            windows_to_delete.append(index)
        
        normalized_coverage_data[index,2:] = (coverage_data[index,2:]) / normalization_factors      # 10000 scaling factor removed
        #print("coverage:")
        #print(coverage_data[index,:])
        #print("normalized coverage:")
        #print(normalized_coverage_data[index,:])

    normalized_coverage_data = np.delete(normalized_coverage_data, windows_to_delete, axis=0)
    normalized_coverage_data_no_window = normalized_coverage_data[:,2:]
    sequence_data = np.delete(sequence_data, windows_to_delete, axis=0)

    return normalized_coverage_data, normalized_coverage_data_no_window, sequence_data

def custom_train_test_split(X, Y, window_size, overlap, test_size, random_state=None):
    
    if random_state is not None:
        np.random.seed(random_state)

    # Calculate the number of samples
    n_samples = X.shape[0]

    # Calculate the number of samples to be used in the test set
    n_test_samples = int(n_samples * test_size)

    # Calculate the safety margin to avoid overlap
    # The safety margin is the number of windows needed to cover the overlap
    safety_margin = int(overlap / (window_size - overlap))

    # Calculate the start and end indexes for the test set
    test_start_index = (n_samples // 2 ) - (n_test_samples // 2) 
    test_end_index = test_start_index + n_test_samples 


    # Split the data into training and testing sets
    X_train = np.concatenate((X[:(test_start_index - safety_margin)], X[(test_end_index + safety_margin):]), axis=0)
    Y_train = np.concatenate((Y[:(test_start_index - safety_margin)], Y[(test_end_index + safety_margin):]), axis=0)
    X_test = X[test_start_index:test_end_index]
    Y_test = Y[test_start_index:test_end_index]

    X_train, Y_train = shuffle(X_train, Y_train, random_state=random_state)
    X_test, Y_test = shuffle(X_test, Y_test, random_state=random_state)

    return X_train, X_test, Y_train, Y_test

def gaussian_smooth_profiles(Y, sigma=2):
    """
    Apply Gaussian smoothing to each coverage profile in Y_train.

    Parameters:
    Y_train (numpy.ndarray): 2D array where each row is a coverage profile.
    sigma (float): Standard deviation for Gaussian kernel.

    Returns:
    numpy.ndarray: Smoothed coverage profiles.
    """
    # Define a lambda function for applying Gaussian smoothing to a single profile
    smooth_func = lambda profile: gaussian_filter1d(profile, sigma=sigma)

    # Apply the smoothing function to each row (coverage profile) in Y_train
    Y_smoothed = Y
    Y_smoothed[:,2:] = np.apply_along_axis(smooth_func, axis=1, arr=Y[:,2:])

    return Y_smoothed