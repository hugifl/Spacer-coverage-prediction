from custom_elements import  poisson_loss, NaNChecker, calculate_pearson_correlation, find_and_plot_peaks, calculate_peak_f1_score, PearsonCorrelationCallback, F1ScoreCallback
from scipy.signal import find_peaks
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import clone_model
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, Dropout, Masking, Bidirectional, LSTM, concatenate


def filter_annotation_features(X_train_anno, X_test_anno, annotation_features_to_use):

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

    X_train_anno_filtered = X_train_anno[..., indices_to_keep]
    X_test_anno_filtered = X_test_anno[..., indices_to_keep]

    return X_train_anno_filtered, X_test_anno_filtered

def filter_annotation_features_TU(X_train_anno, X_test_anno, annotation_features_to_use):

    feature_index_map = {
    'gene_vector': 0,
    'promoter_vector': 1,
    'terminator_vector': 2,
    'gene_directionality_vector': 3
    }

    indices_to_keep = [feature_index_map[feature] for feature in annotation_features_to_use]

    X_train_anno_filtered = X_train_anno[..., indices_to_keep]
    X_test_anno_filtered = X_test_anno[..., indices_to_keep]

    return X_train_anno_filtered, X_test_anno_filtered



def evaluate_model(Y_true, Y_pred, model_name, outdir, dataset_name, pad_symbol, width=10, prominence=100, overlap_threshold=0.02):
    np.random.seed(42)  # Set seed for reproducibility
    random_indices = np.random.choice(Y_true.shape[0], size=10, replace=False)
    
    pearson_correlations = []
    f1_scores = []

    for i in range(Y_true.shape[0]):
        observed = Y_true[i].flatten()
        predicted = Y_pred[i].flatten()

        observed_rounded = np.round(observed, 2)
        pad_symbol_rounded = np.round(pad_symbol, 2)

        # Find the first index where there are at least 3 subsequent pad symbols
        count = 0
        first_pad_index = None
        for i in range(len(observed_rounded)):
            if observed_rounded[i] == pad_symbol_rounded:
                count += 1
                if count >= 20:
                    first_pad_index = i - 22  # Adjust for 3 subsequent pad symbols
                    break
            else:
                count = 0

        # If at least 3 subsequent pad symbols were found, truncate observed and predicted arrays
        if first_pad_index is not None:
            observed = observed[:first_pad_index]
            predicted = predicted[:first_pad_index]
            

        else:
            first_pad_index = len(observed)
        
        observed = observed[:first_pad_index]
        predicted = predicted[:first_pad_index]

        # Calculate Pearson Correlation for each sample
        corr = calculate_pearson_correlation(observed, predicted, use_log=True, plot=False)
        pearson_correlations.append(corr)

        # Identify peaks and calculate F1 Score for each sample
        observed_peaks, observed_properties = find_peaks(observed, width=width, prominence=prominence)
        predicted_peaks, predicted_properties = find_peaks(predicted, width=width, prominence=prominence)

        # Adjusted calculate_peak_f1_score call for sample-wise evaluation
        f1_score = calculate_peak_f1_score(observed_peaks, predicted_peaks, observed_properties, predicted_properties, overlap_threshold=overlap_threshold, data_length=observed.shape[0])
        f1_scores.append(f1_score)
        if i in random_indices:
            plot_profiles_with_peaks(observed, predicted, i, model_name, outdir, dataset_name, width, prominence)

    avg_pearson_correlation = np.mean(pearson_correlations)
    avg_f1_score = np.mean(f1_scores)

    return avg_pearson_correlation, avg_f1_score

def evaluate_model_2_outputs(Y_true, Y_pred, model_name, outdir, dataset_name, pad_symbol, width=10, prominence=100, overlap_threshold=0.02):
    np.random.seed(42)  # Set seed for reproducibility
    random_indices = np.random.choice(Y_true.shape[0], size=10, replace=False)
    
    pearson_correlations = []
    f1_scores = []

    for i in range(Y_true.shape[0]):
        observed = Y_true[i, :, 0].flatten()
        predicted = Y_pred[i, :, 0].flatten()

        observed_rounded = np.round(observed, 4)
        pad_symbol_rounded = np.round(pad_symbol, 4)

        # Find the first index where there are at least 3 subsequent pad symbols
        count = 0
        first_pad_index = None
        for i in range(len(observed_rounded)):
            if observed_rounded[i] == pad_symbol_rounded:
                count += 1
                if count >= 20:
                    first_pad_index = i - 22  # Adjust for 3 subsequent pad symbols
                    break
            else:
                count = 0

        # If at least 3 subsequent pad symbols were found, truncate observed and predicted arrays
        if first_pad_index is not None:
            observed = observed[:first_pad_index]
            predicted = predicted[:first_pad_index]
            

        else:
            first_pad_index = len(observed)
        
        observed = observed[:first_pad_index]
        predicted = predicted[:first_pad_index]

        # Calculate Pearson Correlation for each sample
        corr = calculate_pearson_correlation(observed, predicted, use_log=True, plot=False)
        pearson_correlations.append(corr)

        # Identify peaks and calculate F1 Score for each sample
        observed_peaks, observed_properties = find_peaks(observed, width=width, prominence=prominence)
        predicted_peaks, predicted_properties = find_peaks(predicted, width=width, prominence=prominence)

        # Adjusted calculate_peak_f1_score call for sample-wise evaluation
        f1_score = calculate_peak_f1_score(observed_peaks, predicted_peaks, observed_properties, predicted_properties, overlap_threshold=overlap_threshold, data_length=observed.shape[0])
        f1_scores.append(f1_score)
        if i in random_indices:
            plot_profiles_with_peaks(observed, predicted, i, model_name, outdir, dataset_name, width, prominence)

    avg_pearson_correlation = np.mean(pearson_correlations)
    avg_f1_score = np.mean(f1_scores)

    return avg_pearson_correlation, avg_f1_score


def plot_profiles_with_peaks(observed, predicted, index, model_name, outdir, dataset_name, width, prominence):
    peak_plots_dir = os.path.join(outdir, dataset_name + "_outputs", f"peak_plots_{model_name}")
    os.makedirs(peak_plots_dir, exist_ok=True)  # Ensure the directory exists
    
    # Plot observed
    observed_peaks, observed_properties = find_peaks(observed, width=width, prominence=prominence)
    plt.figure(figsize=(10, 6))
    plt.plot(observed, label='Observed Coverage Profile')
    plt.plot(observed_peaks, observed[observed_peaks], "x", color='red', label='Observed Peaks')
    plt.vlines(x=observed_peaks, ymin=observed[observed_peaks] - observed_properties["prominences"],
               ymax=observed[observed_peaks], color="C1")
    if 'width_heights' in observed_properties:
        plt.hlines(y=observed_properties["width_heights"], xmin=observed_properties["left_ips"],
                   xmax=observed_properties["right_ips"], color="C1")
    plt.xlabel('Position')
    plt.ylabel('Coverage')
    plt.title(f'Observed Coverage Profile with Peaks (Sample {index})')
    plt.legend()
    observed_filename = f'{index}_observed_peaks_{model_name}.png'
    plt.savefig(os.path.join(peak_plots_dir, observed_filename))
    plt.close()

    # Plot predicted
    predicted_peaks, predicted_properties = find_peaks(predicted, width=width, prominence=prominence)
    plt.figure(figsize=(10, 6))
    plt.plot(predicted, label='Predicted Coverage Profile')
    plt.plot(predicted_peaks, predicted[predicted_peaks], "x", color='red', label='Predicted Peaks')
    plt.vlines(x=predicted_peaks, ymin=predicted[predicted_peaks] - predicted_properties["prominences"],
               ymax=predicted[predicted_peaks], color="C1")
    if 'width_heights' in predicted_properties:
        plt.hlines(y=predicted_properties["width_heights"], xmin=predicted_properties["left_ips"],
                   xmax=predicted_properties["right_ips"], color="C1")
    plt.xlabel('Position')
    plt.ylabel('Coverage')
    plt.title(f'Predicted Coverage Profile with Peaks (Sample {index})')
    plt.legend()
    predicted_filename = f'{index}_predicted_peaks_{model_name}.png'
    plt.savefig(os.path.join(peak_plots_dir, predicted_filename))
    plt.close()


def custom_batch_generator(X_seq, X_anno, Y, batch_size=32, max_length_threshold=4000):
    while True:
        # Calculate actual lengths of TUs
        TU_lengths = Y[:, 1] - Y[:, 0]
        
        # Sort TUs by length for batching
        sorted_indices = np.argsort(TU_lengths)
        X_seq_sorted = X_seq[sorted_indices]
        X_anno_sorted = X_anno[sorted_indices]
        Y_sorted = Y[sorted_indices]
        
        # Calculate the number of batches
        num_batches = np.ceil(len(Y_sorted) / batch_size).astype(int)
        # Shuffle the batch order
        batch_order = np.arange(num_batches)
        np.random.shuffle(batch_order)
        
        # Generate batches in shuffled order
        for i in batch_order:
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(Y_sorted))
            
            # Select indices for the current batch and shuffle them
            batch_indices = np.arange(start_idx, end_idx)
            np.random.shuffle(batch_indices)
            
            # Apply shuffled indices to get the batch data
            batch_seq = X_seq_sorted[batch_indices]
            batch_anno = X_anno_sorted[batch_indices]
            batch_Y = Y_sorted[batch_indices]
            
            # Determine the length of the longest TU in the batch for trimming
            max_length = int(max(batch_Y[:, 1] - batch_Y[:, 0]))
           

            # Check if max_length exceeds the threshold
            if max_length > max_length_threshold:
                #print("max_length is ", max_length)
                # Split batch into two equal parts (or as equal as possible)
                mid_point = len(batch_indices) // 2
                #print("start_idx is ", start_idx)
                #print("end_idx is ", end_idx)
                #print("mid_point is ", mid_point)
                for part in [slice(0, mid_point), slice(mid_point, len(batch_indices))]:
                    batch_seq_part = batch_seq[part]
                    batch_anno_part = batch_anno[part]
                    batch_Y_part = batch_Y[part]
                    #print("part is ", part)
                    # Update max_length for the sub-batch
                    max_length_part = int(max(batch_Y_part[:, 1] - batch_Y_part[:, 0]))
                    batch_seq_trimmed = batch_seq_part[:, :max_length_part, :]
                    batch_anno_trimmed = batch_anno_part[:, :max_length_part, :]
                    batch_Y_part_coverage = batch_Y_part[:, 2:]
                    batch_Y_coverage_trimmed = batch_Y_part_coverage[:, :max_length_part]
                    #print("shape of batch_seq_trimmed is ", batch_seq_trimmed.shape)
                    #print("yielding for trimmed ")
                    yield [batch_seq_trimmed, batch_anno_trimmed], batch_Y_coverage_trimmed
            else:
                batch_seq_trimmed = batch_seq[:, :max_length, :]
                batch_anno_trimmed = batch_anno[:, :max_length, :]
                batch_Y_coverage = batch_Y[:, 2:]
                batch_Y_coverage_trimmed = batch_Y_coverage[:, :max_length]
                
                yield [batch_seq_trimmed, batch_anno_trimmed], batch_Y_coverage_trimmed


def transform_sample(sample, pad_symbol):
    # Find the start of the padding sequence
    pad_start_index = None
    for i in range(len(sample) - 4):  # Adjusted to -4 for correct indexing
        if np.all(sample[i:i+5] == pad_symbol):
            pad_start_index = i
            break
    
    # If padding is found, split the transformation
    if pad_start_index is not None:
        non_padded_transformed = np.vstack((sample[:pad_start_index], 1 - sample[:pad_start_index])).T
        padded_transformed = np.full((len(sample) - pad_start_index, 2), pad_symbol)
        return np.vstack((non_padded_transformed, padded_transformed))
    else:
        # If no padding, transform the whole sample
        return np.vstack((sample, 1 - sample)).T

def custom_batch_generator_2_outputs(X_seq, X_anno, Y, pad_symbol, batch_size=32, max_length_threshold=4000):
    while True:
        # Calculate actual lengths of TUs
        TU_lengths = Y[:, 1] - Y[:, 0]
        
        # Sort TUs by length for batching
        sorted_indices = np.argsort(TU_lengths)
        X_seq_sorted = X_seq[sorted_indices]
        X_anno_sorted = X_anno[sorted_indices]
        Y_sorted = Y[sorted_indices]
        
        # Calculate the number of batches
        num_batches = np.ceil(len(Y_sorted) / batch_size).astype(int)
        # Shuffle the batch order
        batch_order = np.arange(num_batches)
        np.random.shuffle(batch_order)
        
        # Generate batches in shuffled order
        for i in batch_order:
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(Y_sorted))
            
            # Select indices for the current batch and shuffle them
            batch_indices = np.arange(start_idx, end_idx)
            np.random.shuffle(batch_indices)
            
            # Apply shuffled indices to get the batch data
            batch_seq = X_seq_sorted[batch_indices]
            batch_anno = X_anno_sorted[batch_indices]
            batch_Y = Y_sorted[batch_indices]
            
            # Determine the length of the longest TU in the batch for trimming
            max_length = int(max(batch_Y[:, 1] - batch_Y[:, 0]))
           

            # Check if max_length exceeds the threshold
            if max_length > max_length_threshold:
                #print("max_length is ", max_length)
                # Split batch into two equal parts (or as equal as possible)
                mid_point = len(batch_indices) // 2
                #print("start_idx is ", start_idx)
                #print("end_idx is ", end_idx)
                #print("mid_point is ", mid_point)
                for part in [slice(0, mid_point), slice(mid_point, len(batch_indices))]:
                    batch_seq_part = batch_seq[part]
                    batch_anno_part = batch_anno[part]
                    batch_Y_part = batch_Y[part]
                    #print("part is ", part)
                    # Update max_length for the sub-batch
                    max_length_part = int(max(batch_Y_part[:, 1] - batch_Y_part[:, 0]))
                    batch_seq_trimmed = batch_seq_part[:, :max_length_part, :]
                    batch_anno_trimmed = batch_anno_part[:, :max_length_part, :]
                    batch_Y_part_coverage = batch_Y_part[:, 2:]
                    batch_Y_coverage_trimmed = batch_Y_part_coverage[:, :max_length_part]
                    final_transformed = np.array([transform_sample(sample, pad_symbol) for sample in batch_Y_coverage_trimmed])
                    #print("yielding for trimmed ")
                    yield [batch_seq_trimmed, batch_anno_trimmed], final_transformed
            else:
                batch_seq_trimmed = batch_seq[:, :max_length, :]
                batch_anno_trimmed = batch_anno[:, :max_length, :]
                batch_Y_coverage = batch_Y[:, 2:]
                batch_Y_coverage_trimmed = batch_Y_coverage[:, :max_length]
                final_transformed = np.array([transform_sample(sample, pad_symbol) for sample in batch_Y_coverage_trimmed])

                yield [batch_seq_trimmed, batch_anno_trimmed], final_transformed

def clip_test_set(X_seq, X_anno, Y):
    TU_lengths = Y[:, 1] - Y[:, 0]  # Calculate actual lengths of TUs
    

    # Trim each TU to the length of the longest TU
    max_length = int(max(TU_lengths))
    X_seq_trimmed = X_seq[:, :max_length, :]
    X_anno_trimmed = X_anno[:, :max_length, :]
    Y_coverage = Y[:, 2:] 
    Y_coverage_trimmed = Y_coverage[:, :max_length]
    
    return X_seq_trimmed, X_anno_trimmed, Y_coverage_trimmed    


def clip_test_set_2_outputs(X_seq, X_anno, Y, pad_symbol):
    TU_lengths = Y[:, 1] - Y[:, 0]  # Calculate actual lengths of TUs
    

    # Trim each TU to the length of the longest TU
    max_length = int(max(TU_lengths))
    X_seq_trimmed = X_seq[:, :max_length, :]
    X_anno_trimmed = X_anno[:, :max_length, :]
    Y_coverage = Y[:, 2:] 
    Y_coverage_trimmed = Y_coverage[:, :max_length]
    final_transformed = np.array([transform_sample(sample, pad_symbol) for sample in Y_coverage_trimmed])

    return X_seq_trimmed, X_anno_trimmed, final_transformed  


def calculate_total_batches(Y, batch_size=32, max_length_threshold=4000):
    TU_lengths = Y[:, 1] - Y[:, 0]
    sorted_indices = np.argsort(TU_lengths)
    Y_sorted = Y[sorted_indices]
    
    total_batches = 0
    for i in range(0, len(Y_sorted), batch_size):
        end_idx = min(i + batch_size, len(Y_sorted))
        batch_Y = Y_sorted[i:end_idx]
        max_length = int(max(batch_Y[:, 1] - batch_Y[:, 0]))
        
        if max_length > max_length_threshold:
            # Assuming each split results in 2 parts
            total_batches += 2
        else:
            total_batches += 1
    
    return total_batches


def restrict_TU_lengths(X_test_filtered, Y_test_filtered, min_length=100, max_length=4000):
    TU_lengths = Y_test_filtered[:, 1] - Y_test_filtered[:, 0]
    mask = (TU_lengths >= min_length) & (TU_lengths <= max_length)
    X_test_filtered_restricted = X_test_filtered[mask]
    Y_test_filtered_restricted = Y_test_filtered[mask]
    
    return X_test_filtered_restricted, Y_test_filtered_restricted


def add_missing_padding(X_train, pad_symbol):
    # Identify the positions where any of the 8 dimensions have the pad_symbol
    pad_positions = np.any(X_train == pad_symbol, axis=2)
    
    # For each position identified, set the entire 8 dimensions to pad_symbol
    for i in range(X_train.shape[0]):
        for j in range(X_train.shape[1]):
            if pad_positions[i, j]:
                X_train[i, j, :] = pad_symbol
    return X_train


def split_data_by_length(X_seq, X_anno, Y, N=3):
    # Calculate actual lengths of TUs
    TU_lengths = Y[:, 1] - Y[:, 0]
    
    # Find the N-1 quantile values to split at
    quantiles = np.quantile(TU_lengths, np.linspace(0, 1, N+1))
    
    # print mean length of TUs in each quantile
    mean_lengths = []
    for i in range(N):
        indices = np.where((TU_lengths >= quantiles[i]) & (TU_lengths < quantiles[i+1]))[0]
        mean_length = np.mean(TU_lengths[indices])
        mean_lengths.append(mean_length)
        print(f"Mean length of TUs in quantile {i+1}: {mean_length}")

    # Split data into N groups based on these quantiles
    sub_datasets = []
    for i in range(N):
        indices = np.where((TU_lengths >= quantiles[i]) & (TU_lengths < quantiles[i+1]))[0]
        sub_datasets.append((X_seq[indices], X_anno[indices], Y[indices]))
    
    return sub_datasets

def copy_modeld(original_model, N=3, custom_objects=None):
    models = []
    for _ in range(N):
        # Create a new instance of the model class
        model_copy = original_model.__class__(
            # Add all the original initialization parameters here
            CNN_num_layers_seq=original_model.CNN_num_layers_seq,
            CNN_num_layers_anno=original_model.CNN_num_layers_anno,
            filter_number_seq=original_model.filter_number_seq,
            filter_number_anno=original_model.filter_number_anno,
            kernel_size_seq=original_model.kernel_size_seq,
            kernel_size_anno=original_model.kernel_size_anno,
            biLSTM_num_layers_seq=original_model.biLSTM_num_layers_seq,
            biLSTM_num_layers_anno=original_model.biLSTM_num_layers_anno,
            unit_numbers_seq=original_model.unit_numbers_seq,
            unit_numbers_anno=original_model.unit_numbers_anno,
            unit_numbers_combined=original_model.unit_numbers_combined,
            only_seq=original_model.only_seq,
            pad_symbol=original_model.pad_symbol
        )
        
        # Ensure the model is built. In some cases, you might need to provide an input shape.
        model_copy.build((None, ) + original_model.input_shape[1:])
        
        # Copy the weights from the original model
        model_copy.set_weights(original_model.get_weights())
        
        # Compile the model if necessary
        if custom_objects:
            model_copy.compile(optimizer=original_model.optimizer,
                               loss=original_model.loss,
                               metrics=original_model.metrics,
                               loss_weights=original_model.loss_weights,
                               weighted_metrics=original_model.weighted_metrics,
                               run_eagerly=original_model.run_eagerly,
                               **custom_objects)
        else:
            model_copy.compile(optimizer=original_model.optimizer,
                               loss=original_model.loss,
                               metrics=original_model.metrics,
                               loss_weights=original_model.loss_weights,
                               weighted_metrics=original_model.weighted_metrics,
                               run_eagerly=original_model.run_eagerly)
        
        models.append(model_copy)
    return models


def copy_model(original_model, N=3, dummy_input_shapes=((1, 100, 4), (1, 100, 4))):
    copied_models = []
    for _ in range(N):
        # Initialize a new model instance with the same configuration as the original model
        model_copy = type(original_model)(
            CNN_num_layers_seq=original_model.CNN_num_layers_seq,
            CNN_num_layers_anno=original_model.CNN_num_layers_anno,
            filter_number_seq=original_model.filter_number_seq,
            filter_number_anno=original_model.filter_number_anno,
            kernel_size_seq=original_model.kernel_size_seq,
            kernel_size_anno=original_model.kernel_size_anno,
            biLSTM_num_layers_seq=original_model.biLSTM_num_layers_seq,
            biLSTM_num_layers_anno=original_model.biLSTM_num_layers_anno,
            unit_numbers_seq=original_model.unit_numbers_seq,
            unit_numbers_anno=original_model.unit_numbers_anno,
            unit_numbers_combined=original_model.unit_numbers_combined,
            only_seq=original_model.only_seq,
            pad_symbol=original_model.pad_symbol
        )

        # Generate dummy inputs for both the sequence and annotation streams
        dummy_input_seq = np.random.randn(*dummy_input_shapes[0]).astype(np.float32)
        dummy_input_anno = np.random.randn(*dummy_input_shapes[1]).astype(np.float32)

        # Perform a dummy forward pass to build the model
        model_copy([tf.convert_to_tensor(dummy_input_seq), tf.convert_to_tensor(dummy_input_anno)])

        # Copy the weights from the original model
        model_copy.set_weights(original_model.get_weights())
        copied_models.append(model_copy)
    return copied_models