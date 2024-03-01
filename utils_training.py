from custom_elements import  poisson_loss, NaNChecker, calculate_pearson_correlation, find_and_plot_peaks, calculate_peak_f1_score, PearsonCorrelationCallback, F1ScoreCallback
from scipy.signal import find_peaks
import numpy as np
import os
import matplotlib.pyplot as plt


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





def evaluate_model(Y_true, Y_pred, model_name, outdir, dataset_name, width=10, prominence=100, overlap_threshold=0.02):
    np.random.seed(42)  # Set seed for reproducibility
    random_indices = np.random.choice(Y_true.shape[0], size=10, replace=False)
    
    pearson_correlations = []
    f1_scores = []

    for i in range(Y_true.shape[0]):
        observed = Y_true[i].flatten()
        predicted = Y_pred[i].flatten()

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

def custom_batch_generator(X_seq, X_anno, Y, batch_size=32):
    # Assuming Y[:, 0:2] contains the Start and End indices for TUs
    TU_lengths = Y[:, 1] - Y[:, 0]  # Calculate actual lengths of TUs
    
    # Sort TUs by length for batching
    sorted_indices = np.argsort(TU_lengths)
    X_seq_sorted = X_seq[sorted_indices]
    X_anno_sorted = X_anno[sorted_indices]
    Y_sorted = Y[sorted_indices]
    
    # Generate batches
    for start_idx in range(0, len(Y_sorted), batch_size):
        end_idx = min(start_idx + batch_size, len(Y_sorted))
        batch_seq = X_seq_sorted[start_idx:end_idx]
        batch_anno = X_anno_sorted[start_idx:end_idx]
        batch_Y = Y_sorted[start_idx:end_idx]
        
        # Trim each TU in the batch to the length of the longest TU
        max_length = max(batch_Y[:, 1] - batch_Y[:, 0])
        batch_seq_trimmed = batch_seq[:, :max_length, :]
        batch_anno_trimmed = batch_anno[:, :max_length, :]
        
        yield batch_seq_trimmed, batch_anno_trimmed, batch_Y