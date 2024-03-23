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

def custom_train_test_split_TU(X, Y, test_size, random_state=None):
    
    if random_state is not None:
        np.random.seed(random_state)

    # Calculate the number of samples
    n_samples = X.shape[0]

    # Calculate the number of samples to be used in the test set
    n_test_samples = int(n_samples * test_size)


    # Calculate the start and end indexes for the test set
    test_start_index = (n_samples // 2 ) - (n_test_samples // 2) 
    test_end_index = test_start_index + n_test_samples 


    # Split the data into training and testing sets
    X_train = np.concatenate((X[:(test_start_index)], X[(test_end_index):]), axis=0)
    Y_train = np.concatenate((Y[:(test_start_index)], Y[(test_end_index):]), axis=0)
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


def scale_to_0_1(Y, pad_symbol=0.42):
    pad_symbol = round(pad_symbol, 2) 
    Y = np.round(Y, 2)
    """
    Scale the coverage profiles to the range [0, 1], stopping at the first occurrence
    of at least 5 repetitions of the pad_symbol.

    Parameters:
    Y (numpy.ndarray): 2D array where each row is a coverage profile.
    pad_symbol (float): The symbol used for padding.

    Returns:
    numpy.ndarray: Scaled coverage profiles.
    """
    def find_pad_index(profile, pad_symbol):
        """
        Find the index where at least 5 consecutive pad_symbol starts.

        Parameters:
        profile (numpy.ndarray): 1D array representing a single coverage profile.
        pad_symbol (float): The symbol used for padding.

        Returns:
        int: The index where padding starts, or the length of the profile if no such padding exists.
        """
        # Convert to a list for easier processing
        profile_list = profile.tolist()
        # Generate a string representation for easy pattern search
        pattern = [pad_symbol] * 10
        for i in range(len(profile_list)):
            if profile_list[i:i+10] == pattern:
                return i
        return len(profile_list)

    def scale_profile(profile):
        """
        Scale the profile to the range [0, 1].

        Parameters:
        profile (numpy.ndarray): The profile to scale.

        Returns:
        numpy.ndarray: The scaled profile.
        """
        if np.min(profile) == np.max(profile):  # Handle division by zero if all values are the same
            return np.zeros_like(profile)
        else:
            return (profile - np.min(profile)) / (np.max(profile) - np.min(profile))

    Y_scaled = np.zeros_like(Y)
    for i, row in enumerate(Y):
        pad_index = find_pad_index(row, pad_symbol)
        # Scale only the non-pad part if pad_index is found; otherwise, scale the entire row
        if pad_index > 0:  # Check to ensure there's data to scale
            Y_scaled[i, :pad_index] = scale_profile(row[:pad_index])
            if pad_index < len(row):  # If there's padding, fill the rest with the pad symbol
                Y_scaled[i, pad_index:] = pad_symbol
        else:  # If no pad_symbol sequence is found, scale the entire row
            Y_scaled[i] = scale_profile(row)

    return Y_scaled



def scale_to_0_1_global(Y, pad_symbol=0.42):
    pad_symbol = round(pad_symbol, 2)
    Y = np.round(Y, 2)
    """
    Scale the coverage profiles to the range [0, 1] based on global min and max values,
    stopping at the first occurrence of at least 5 consecutive pad_symbol.
    Only scales values before padding.

    Parameters:
    Y (numpy.ndarray): 2D array where each row is a coverage profile.
    pad_symbol (float): The symbol used for padding.

    Returns:
    numpy.ndarray: Scaled coverage profiles with padding preserved.
    """
    def find_pad_start(profile, pad_symbol):
        """
        Find the start index of at least 5 consecutive pad_symbol values.

        Parameters:
        profile (numpy.ndarray): 1D array representing a single coverage profile.
        pad_symbol (float): The symbol used for padding.

        Returns:
        int: Index where padding starts, or None if padding not found.
        """
        for i in range(len(profile) - 10):  # -4 to leave space for 5 consecutive checks
            if all(profile[i:i+10] == pad_symbol):
                return i
        return None

    # Identify non-padding parts and calculate global min and max from them
    non_padding_parts = []
    for row in Y:
        pad_start_index = find_pad_start(row, pad_symbol)
        if pad_start_index is not None:
            non_padding_parts.append(row[:pad_start_index])
        else:
            non_padding_parts.append(row)
    non_padding_values = np.concatenate(non_padding_parts)
    global_min = non_padding_values.min()
    global_max = non_padding_values.max()

    # Apply global scaling to the non-padding part of each profile
    Y_scaled = np.zeros_like(Y)
    for i, row in enumerate(Y):
        pad_start_index = find_pad_start(row, pad_symbol)
        if pad_start_index is not None:
            Y_scaled[i, :pad_start_index-5] = (row[:pad_start_index-5] - global_min) / (global_max - global_min)
            Y_scaled[i, pad_start_index-5:] = pad_symbol  # Preserve padding
        else:
            Y_scaled[i] = (row - global_min) / (global_max - global_min)

    return Y_scaled


def scale_to_0_1_global_with_max(Y, pad_symbol=0.42, max_value=100):
    pad_symbol = round(pad_symbol, 2)
    Y_rounded = np.round(Y, 2)
    """
    Scale the coverage profiles to the range [0, 1], where max_value defines the scaled value of 1.
    Values greater than max_value are set to 1. Padding is preserved.

    Parameters:
    Y (numpy.ndarray): 2D array where each row is a coverage profile.
    pad_symbol (float): The symbol used for padding.
    max_value (float): The value that will be scaled to 1.

    Returns:
    numpy.ndarray: Scaled coverage profiles with padding preserved.
    """
    def find_pad_start(profile, pad_symbol):
        for i in range(len(profile) - 14):
            if all(profile[i:i+15] == pad_symbol):
                return i
        return None

    # Cap the values at max_value
    Y_capped = np.minimum(Y, max_value)

    # Exclude padding for global min calculation
    mask = Y_rounded != pad_symbol
    global_min = Y_capped[mask].min()
    global_max = Y[mask].max()
    print("global max: ", global_max)

    # Apply scaling based on capped values
    Y_scaled = np.zeros_like(Y)

    for i, row in enumerate(Y):
        pad_start_index = find_pad_start(row, pad_symbol)
        if pad_start_index is not None:
            # Scale non-padding values
            scaled_row = (row[:pad_start_index+5] ) / max_value
            Y_scaled[i, :pad_start_index+5] = np.clip(scaled_row, 0, 1)  # Ensure values are within [0, 1]
            Y_scaled[i, pad_start_index+5:] = pad_symbol  # Preserve padding
        else:
            # Scale entire row if no padding is detected
            scaled_row = (row ) / max_value
            Y_scaled[i] = np.clip(scaled_row, 0, 1)

    return Y_scaled