import tensorflow as tf
import scipy
import numpy
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd
from tensorflow.keras.callbacks import Callback
from scipy.signal import find_peaks
import numpy as np
from tensorflow.keras.layers import Layer, Dense, Flatten, Activation, RepeatVector, Permute, Multiply, Lambda
import tensorflow.keras.backend as K
import random


# Pooling layer that applies max pooling on all channels and max pooling of the absolute values on the last channel (contains -1/1 values)
class CustomPooling(tf.keras.layers.Layer):
    def __init__(self, pool_size, strides, padding='SAME', **kwargs):
        super(CustomPooling, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.large_positive_value = tf.constant(2.0, dtype=tf.float32) 

    def call(self, inputs):
        # Separate the last channel
        last_channel = inputs[..., -1:]

        # Replace -1 with a large positive value
        replaced_last_channel = tf.where(last_channel == -1, self.large_positive_value, last_channel)

        # Apply max pooling on all channels except the last
        pooled = tf.nn.max_pool(inputs[..., :-1], ksize=[1, self.pool_size, 1], strides=[1, self.strides, 1], padding=self.padding)

        # Apply max pooling on the last channel with replaced values
        pooled_last_channel = tf.nn.max_pool(replaced_last_channel, ksize=[1, self.pool_size, 1], strides=[1, self.strides, 1], padding=self.padding)

        # Restore the original sign in the last channel
        restored__last_channel = tf.where(pooled_last_channel == self.large_positive_value, tf.constant(-1.0, dtype=tf.float32), pooled_last_channel)

        # Concatenate pooled channels and the last channel
        output = tf.concat([pooled, restored__last_channel], axis=-1)
        return output
    
# Custom binary cross entropy loss function for binary peak prediction that punishes false positives more than false negatives and does l1 regularization.
def custom_loss_with_l1(y_true, y_pred):
    y_true = tf.cast(y_true, y_pred.dtype)

    # Manually calculate BCE for each position
    bce = -(y_true * tf.math.log(y_pred + 1e-15) + (1 - y_true) * tf.math.log(1 - y_pred + 1e-15))

    # Define weight for false negatives
    false_negative_weight = 5.0

    # Apply weights
    weight = tf.where(tf.less(y_pred, y_true), false_negative_weight, 1.0)
    weighted_bce = bce * weight

    # Sum up the weighted BCEs for each sample in the batch
    weighted_bce_sum = tf.reduce_sum(weighted_bce, axis=-1)

    # L1 Regularization
    l1_lambda = 0.01
    l1_reg = l1_lambda * tf.reduce_sum(tf.abs(y_pred))

    # Combine BCE and L1 regularization
    combined_loss = tf.reduce_mean(weighted_bce_sum) + l1_reg

    return combined_loss




class Attention(Layer):
    def __init__(self, return_sequences=True):
        super(Attention, self).__init__()
        self.return_sequences = return_sequences

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        if self.return_sequences:
            return output
        else:
            return K.sum(output, axis=1)

    def get_config(self):
        config = super(Attention, self).get_config()
        config.update({"return_sequences": self.return_sequences})
        return config

def binary_crossentropy_with_masking(y_true, y_pred):
    pad_symbol = 0.42  # Assuming pad_symbol is 0; adjust as necessary
    
    # Create a mask for non-padded values
    mask = K.cast(K.not_equal(y_true, pad_symbol), K.floatx())
    
    # Calculate binary crossentropy
    loss = -(y_true * K.log(K.clip(y_pred, K.epsilon(), 1 - K.epsilon())) +
             (1 - y_true) * K.log(K.clip(1 - y_pred, K.epsilon(), 1 - K.epsilon())))
    
    # Apply mask to ignore padded values in loss calculation
    masked_loss = loss * mask
    
    # Calculate the total number of positions and non-padded positions
    total_positions = tf.size(mask)
    non_padded_positions = K.sum(mask)
    
    # Correct the type mismatch for subtraction
    padded_positions = total_positions - K.cast(non_padded_positions, 'int32')
    
    # Calculate and return the mean loss, considering only the non-padded values
    return K.sum(masked_loss) / non_padded_positions

# Custom poisson loss function that avoids NaNs and Infs
def poisson_loss(y_true, y_pred):
    return K.mean(K.maximum(.0, y_pred) - y_true * K.log(K.maximum(.0, y_pred) + K.epsilon()), axis=-1)

def poisson_loss_with_masking(y_true, y_pred):
    # Assuming pad_symbol is 0; adjust this as necessary for your application
    pad_symbol = 0.42
    # Create a mask for non-padded values
    mask = K.cast(K.not_equal(y_true, pad_symbol), K.floatx())
   

    # Calculate Poisson loss
    loss = K.maximum(.0, y_pred) - y_true * K.log(K.maximum(.0, y_pred) + K.epsilon())
    
    # Apply mask to ignore padded values in loss calculation
    masked_loss = loss * mask

    total_positions = tf.size(mask)
    non_padded_positions = K.sum(mask)
    padded_positions = total_positions - K.cast(non_padded_positions, 'int32')
    
    # Return the mean loss, considering only the non-padded values
    return K.sum(masked_loss) / K.sum(mask)

class NaNChecker(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        if numpy.isnan(logs['loss']) or numpy.isinf(logs['loss']):
            print(f"Batch {batch}: Invalid loss, terminating training")
            self.model.stop_training = True
    

def spearman_correlation(y_true, y_pred):
    return tf.py_function(lambda a, b: scipy.stats.spearmanr(a, b).correlation, [y_true, y_pred], tf.double)

def calculate_peak_f1_score(observed_peaks, predicted_peaks, observed_properties, predicted_properties, overlap_threshold=0.02, data_length=None):
    
    def check_overlap(peak1_index, properties1, peak2_index, properties2, data_length):
        """ Check if two peaks overlap by at least the overlap_threshold. """
        start1 = int(round(properties1['left_ips'][peak1_index]))
        end1 = int(round(properties1['right_ips'][peak1_index]))
        start2 = int(round(properties2['left_ips'][peak2_index]))
        end2 = int(round(properties2['right_ips'][peak2_index]))

        
        # Ensure the indices are within the valid range
        start1, end1, start2, end2 = max(0, start1), min(end1, data_length), max(0, start2), min(end2, data_length)

        # Calculate the overlap
        overlap = max(0, min(end1, end2) - max(start1, start2))
        
    
        return overlap / max(end1 - start1, end2 - start2) >= overlap_threshold
    
    # Counters for TP, FP, and FN
    tp = fp = fn = 0

    # Check each predicted peak for TP or FP
    for i, predicted_peak in enumerate(predicted_peaks):
        if any(check_overlap(i, predicted_properties, j, observed_properties, data_length) for j, observed_peak in enumerate(observed_peaks)):
            tp += 1
        else:
            fp += 1

    # Check each observed peak for FN
    for j, observed_peak in enumerate(observed_peaks):
        if not any(check_overlap(j, observed_properties, i, predicted_properties, data_length) for i, predicted_peak in enumerate(predicted_peaks)):
            fn += 1

    # Calculate Precision, Recall, and F1 score
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return f1_score

def calculate_pearson_correlation(observed, predicted, use_log=True, plot=True, plot_filename=None):
    if use_log:
        # Apply log2 transformation, adding a small constant to avoid log(0)
        observed_transformed = numpy.log2(observed + 1e-9)
        predicted_transformed = numpy.log2(predicted + 1e-9)
    else:
        observed_transformed = observed
        predicted_transformed = predicted

    correlation, _ = pearsonr(observed_transformed, predicted_transformed)

    if plot:
        plt.figure(figsize=(8, 6))
        plt.scatter(observed_transformed, predicted_transformed, alpha=0.6)
        plt.xlabel("Observed Values" + (" (log2)" if use_log else ""))
        plt.ylabel("Predicted Values" + (" (log2)" if use_log else ""))
        plt.title("Observed vs Predicted Coverage")
        plt.grid(True)
        plt.savefig(plot_filename)
        plt.show()
    return correlation

def find_and_plot_peaks(data, filename, height=None, distance=None, prominence=None, width=None):
    
    # Identifying peaks
    peaks, properties = find_peaks(data, height=height, distance=distance, prominence=prominence, width=width)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(data, label='Coverage Profile')
    plt.plot(peaks, data[peaks], "x", color='red', label='Peaks')
    # Marking the prominence of each peak
    plt.vlines(x=peaks, ymin=data[peaks] - properties["prominences"],
               ymax=data[peaks], color="C1")

    # Marking the width of each peak
    if 'width_heights' in properties and 'left_ips' in properties and 'right_ips' in properties:
        plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"],
                   xmax=properties["right_ips"], color="C1")

    plt.xlabel('Position')
    plt.ylabel('Coverage')
    plt.title('Coverage Profile with Identified Peaks')
    plt.legend()
    plt.savefig(filename)
    plt.show()

    return peaks



class PearsonCorrelationCallback(Callback):
    def __init__(self, X_seq, X_anno, Y_true, batch_size, use_log=True, plot=False):
        super().__init__()
        self.X_seq = X_seq
        self.X_anno = X_anno
        self.Y_true = Y_true
        self.batch_size = batch_size
        self.use_log = use_log
        self.plot = plot
    
    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.predict([self.X_seq, self.X_anno], batch_size=self.batch_size)
        correlations = []
        
        # Iterate through each sample
        for observed, predicted in zip(self.Y_true, predictions):
            if self.use_log:
                observed_transformed = np.log2(observed + 1e-9)
                predicted_transformed = np.log2(predicted + 1e-9)
            else:
                observed_transformed = observed
                predicted_transformed = predicted
            
            # Compute Pearson correlation for each sample
            correlation, _ = pearsonr(observed_transformed, predicted_transformed)
            correlations.append(correlation)
        
        # Calculate the average Pearson correlation
        avg_correlation = np.mean(correlations)
        max_correlation = np.max(correlations)
        min_correlation = np.min(correlations)
        print(f'Epoch {epoch+1} Average Pearson Correlation: {avg_correlation:.4f} Max Pearson Correlation: {max_correlation:.4f} Min Pearson Correlation: {min_correlation:.4f}')

class F1ScoreCallback(Callback):
    def __init__(self, X_seq, X_anno, Y_true, batch_size, width=None, prominence=None, overlap_threshold=0.02, data_length=None):
        super().__init__()
        self.X_seq = X_seq
        self.X_anno = X_anno
        self.Y_true = Y_true
        self.batch_size = batch_size
        self.width = width
        self.prominence = prominence
        self.overlap_threshold = overlap_threshold
        self.data_length = data_length

    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.predict([self.X_seq, self.X_anno], batch_size=self.batch_size)
        f1_scores = []

        # Iterate through each sample in batch
        for observed, predicted in zip(self.Y_true, predictions):
            # Make sure to process one sample at a time, flattening if necessary
            observed = observed.flatten()
            predicted = predicted.flatten()

            observed_peaks, observed_properties = find_peaks(observed, width=self.width, prominence=self.prominence)
            predicted_peaks, predicted_properties = find_peaks(predicted, width=self.width, prominence=self.prominence)

            f1_score = calculate_peak_f1_score(observed_peaks, predicted_peaks, observed_properties, predicted_properties, overlap_threshold=self.overlap_threshold, data_length=self.data_length or observed.shape[0])
            f1_scores.append(f1_score)

        avg_f1_score = np.mean(f1_scores)
        print(f'Epoch {epoch+1} Average F1 Score: {avg_f1_score:.4f}')


class PearsonCorrelationCallback_TU_old(Callback):
    def __init__(self, generator, num_samples=8, use_log=True, plot=False):
        super().__init__()
        self.generator = generator
        self.num_samples = num_samples
        self.use_log = use_log
        self.plot = plot
        self.batch_samples = []

    def on_epoch_begin(self, epoch, logs=None):
        self.batch_samples = []  # Reset at the beginning of each epoch
        # Collect a specified number of batches upfront
        for _, batch in zip(range(self.num_samples), self.generator):
            self.batch_samples.append(batch)

    def on_epoch_end(self, epoch, logs=None):
        correlations = []
        
        # Ensure that there are enough samples to sample from
        num_samples_to_draw = min(self.num_samples, len(self.batch_samples))
        
        # Use random.sample to directly sample batches from self.batch_samples
        sampled_batches = random.sample(self.batch_samples, num_samples_to_draw)
        
        for (X_seq, X_anno), Y_true in sampled_batches:
            # Assuming model.predict can handle the list of inputs directly as your model expects
            predictions = self.model.predict([X_seq, X_anno])
            for observed, predicted in zip(Y_true, predictions):
                if self.use_log:
                    observed_transformed = np.log2(observed + 1e-9)
                    predicted_transformed = np.log2(predicted + 1e-9)
                else:
                    observed_transformed = observed
                    predicted_transformed = predicted
                
                correlation, _ = pearsonr(observed_transformed.flatten(), predicted_transformed.flatten())
                correlations.append(correlation)
        
        avg_correlation = np.mean(correlations)
        print(f'Epoch {epoch+1}: Average Pearson Correlation: {avg_correlation:.4f}')

class PearsonCorrelationCallback_TU(Callback):
    def __init__(self, generator, num_samples=8, use_log=True, plot=False, pad_symbol=0):
        super().__init__()
        self.generator = generator
        self.num_samples = num_samples
        self.use_log = use_log
        self.plot = plot
        self.pad_symbol = pad_symbol
        self.batch_samples = []

    def on_epoch_begin(self, epoch, logs=None):
        self.batch_samples = []  # Reset at the beginning of each epoch
        # Collect a specified number of batches upfront
        for _, batch in zip(range(self.num_samples), self.generator):
            self.batch_samples.append(batch)

    def on_epoch_end(self, epoch, logs=None):
        correlations = []
        
        # Ensure that there are enough samples to sample from
        num_samples_to_draw = min(self.num_samples, len(self.batch_samples))
        
        # Use random.sample to directly sample batches from self.batch_samples
        sampled_batches = random.sample(self.batch_samples, num_samples_to_draw)
        
        for (X_seq, X_anno), Y_true in sampled_batches:
            # Assuming model.predict can handle the list of inputs directly as your model expects
            predictions = self.model.predict([X_seq, X_anno])
            for observed, predicted in zip(Y_true, predictions):
                observed = observed.flatten()
                predicted = predicted.flatten()
                
                #print("dimension of predicted", predicted.shape)
                #print("observed shape: ", observed.shape)
                observed_rounded = np.round(observed, 2)
                pad_symbol_rounded = np.round(self.pad_symbol, 2)

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

                if first_pad_index is not None and first_pad_index < 5:
                    # skip the sample if the length of the sequence is too short
                    continue

                if self.use_log:
                    observed_transformed = np.log2(observed + 1e-9)
                    predicted_transformed = np.log2(predicted + 1e-9)
                else:
                    observed_transformed = observed
                    predicted_transformed = predicted
                
                correlation, _ = pearsonr(observed_transformed, predicted_transformed)
                correlations.append(correlation)
        
        avg_correlation = np.mean(correlations)
        print(f'Epoch {epoch+1}: Average Pearson Correlation: {avg_correlation:.4f}')

class PearsonCorrelationCallback_TU_2_outputs(Callback):
    def __init__(self, generator, num_samples=8, use_log=True, plot=False, pad_symbol=0):
        super().__init__()
        self.generator = generator
        self.num_samples = num_samples
        self.use_log = use_log
        self.plot = plot
        self.pad_symbol = pad_symbol
        self.batch_samples = []

    def on_epoch_begin(self, epoch, logs=None):
        self.batch_samples = []  # Reset at the beginning of each epoch
        # Collect a specified number of batches upfront
        for _, batch in zip(range(self.num_samples), self.generator):
            self.batch_samples.append(batch)

    def on_epoch_end(self, epoch, logs=None):
        correlations = []
        
        # Ensure that there are enough samples to sample from
        num_samples_to_draw = min(self.num_samples, len(self.batch_samples))
        
        # Use random.sample to directly sample batches from self.batch_samples
        sampled_batches = random.sample(self.batch_samples, num_samples_to_draw)
        
        for (X_seq, X_anno), Y_true in sampled_batches:
            # Assuming model.predict can handle the list of inputs directly as your model expects
            predictions = self.model.predict([X_seq, X_anno])
            for observed, predicted in zip(Y_true, predictions):
                observed = observed[0,:].flatten()
                predicted = predicted[0,:].flatten()
                
                #print("dimension of predicted", predicted.shape)
                #print("observed shape: ", observed.shape)
                observed_rounded = np.round(observed, 4)
                pad_symbol_rounded = np.round(self.pad_symbol, 4)

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

                if first_pad_index is not None and first_pad_index < 5:
                    # skip the sample if the length of the sequence is too short
                    continue

                if self.use_log:
                    observed_transformed = np.log2(observed + 1e-9)
                    predicted_transformed = np.log2(predicted + 1e-9)
                else:
                    observed_transformed = observed
                    predicted_transformed = predicted
                
                correlation, _ = pearsonr(observed_transformed, predicted_transformed)
                correlations.append(correlation)
        
        avg_correlation = np.mean(correlations)
        print(f'Epoch {epoch+1}: Average Pearson Correlation: {avg_correlation:.4f}')

class F1ScoreCallback_TU_old(Callback):
    def __init__(self, generator, num_samples=8,width=None, prominence=None, overlap_threshold=0.02):
        super().__init__()
        self.generator = generator
        self.num_samples = num_samples
        self.batch_samples = []
        self.width = width
        self.prominence = prominence
        self.overlap_threshold = overlap_threshold

    def on_epoch_begin(self, epoch, logs=None):
        self.batch_samples = []  # Reset at the beginning of each epoch
        # Collect a specified number of batches upfront
        for _, batch in zip(range(self.num_samples), self.generator):
            self.batch_samples.append(batch)

    def on_epoch_end(self, epoch, logs=None):
        f1_scores = []
        
        # Ensure that there are enough samples to sample from
        num_samples_to_draw = min(self.num_samples, len(self.batch_samples))
        
        # Use random.sample to directly sample batches from self.batch_samples
        sampled_batches = random.sample(self.batch_samples, num_samples_to_draw)
        
        for (X_seq, X_anno), Y_true in sampled_batches:
            # Assuming model.predict can handle the list of inputs directly as your model expects
            predictions = self.model.predict([X_seq, X_anno])
            for observed, predicted in zip(Y_true, predictions):
                observed = observed.flatten()
                predicted = predicted.flatten()

                observed_peaks, observed_properties = find_peaks(observed, width=self.width, prominence=self.prominence)
                predicted_peaks, predicted_properties = find_peaks(predicted, width=self.width, prominence=self.prominence)

                f1_score = calculate_peak_f1_score(observed_peaks, predicted_peaks, observed_properties, predicted_properties, overlap_threshold=self.overlap_threshold, data_length= observed.shape[0])
                f1_scores.append(f1_score)

        avg_f1_score = np.mean(f1_scores)
        print(f'Epoch {epoch+1} Average F1 Score: {avg_f1_score:.4f}')


class F1ScoreCallback_TU(Callback):
    def __init__(self, generator, num_samples=8, width=None, prominence=None, overlap_threshold=0.02, pad_symbol=0):
        super().__init__()
        self.generator = generator
        self.num_samples = num_samples
        self.width = width
        self.prominence = prominence
        self.overlap_threshold = overlap_threshold
        self.pad_symbol = pad_symbol
        self.batch_samples = []

    def on_epoch_begin(self, epoch, logs=None):
        self.batch_samples = []  # Reset at the beginning of each epoch
        for _, batch in zip(range(self.num_samples), self.generator):
            self.batch_samples.append(batch)

    def on_epoch_end(self, epoch, logs=None):
        f1_scores = []
        num_samples_to_draw = min(self.num_samples, len(self.batch_samples))
        sampled_batches = random.sample(self.batch_samples, num_samples_to_draw)
        for (X_seq, X_anno), Y_true in sampled_batches:
            #print("shape of Y_true: ", Y_true.shape)
            predictions = self.model.predict([X_seq, X_anno])
            for observed, predicted in zip(Y_true, predictions):
                observed = observed.flatten()
                predicted = predicted.flatten()
                #print("dimension of predicted", predicted.shape)
                #print("observed shape: ", observed.shape)
                observed_rounded = np.round(observed, 2)
                pad_symbol_rounded = np.round(self.pad_symbol, 2)

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
                
                #print("dimension of predicted after cutting", predicted.shape)
                #print("observed shape after cutting: ", observed.shape)
                
                observed_peaks, observed_properties = find_peaks(observed, width=self.width, prominence=self.prominence)
                predicted_peaks, predicted_properties = find_peaks(predicted, width=self.width, prominence=self.prominence)
                
                f1_score = calculate_peak_f1_score(observed_peaks, predicted_peaks, observed_properties, predicted_properties, overlap_threshold=self.overlap_threshold, data_length=first_pad_index)
                f1_scores.append(f1_score)

        avg_f1_score = np.mean(f1_scores)
        print(f'Epoch {epoch+1} Average F1 Score: {avg_f1_score:.4f}')

class F1ScoreCallback_TU_2_outputs(Callback):
    def __init__(self, generator, num_samples=8, width=None, prominence=None, overlap_threshold=0.02, pad_symbol=0):
        super().__init__()
        self.generator = generator
        self.num_samples = num_samples
        self.width = width
        self.prominence = prominence
        self.overlap_threshold = overlap_threshold
        self.pad_symbol = pad_symbol
        self.batch_samples = []

    def on_epoch_begin(self, epoch, logs=None):
        self.batch_samples = []  # Reset at the beginning of each epoch
        for _, batch in zip(range(self.num_samples), self.generator):
            self.batch_samples.append(batch)

    def on_epoch_end(self, epoch, logs=None):
        f1_scores = []
        num_samples_to_draw = min(self.num_samples, len(self.batch_samples))
        sampled_batches = random.sample(self.batch_samples, num_samples_to_draw)
        for (X_seq, X_anno), Y_true in sampled_batches:
            #print("shape of Y_true: ", Y_true.shape)
            predictions = self.model.predict([X_seq, X_anno])
            for observed, predicted in zip(Y_true, predictions):
                observed = observed[0,:].flatten()
                predicted = predicted[0,:].flatten()
                #print("dimension of predicted", predicted.shape)
                #print("observed shape: ", observed.shape)
                observed_rounded = np.round(observed, 4)
                pad_symbol_rounded = np.round(self.pad_symbol, 4)

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
                
                #print("dimension of predicted after cutting", predicted.shape)
                #print("observed shape after cutting: ", observed.shape)
                
                observed_peaks, observed_properties = find_peaks(observed, width=self.width, prominence=self.prominence)
                predicted_peaks, predicted_properties = find_peaks(predicted, width=self.width, prominence=self.prominence)
                
                f1_score = calculate_peak_f1_score(observed_peaks, predicted_peaks, observed_properties, predicted_properties, overlap_threshold=self.overlap_threshold, data_length=first_pad_index)
                f1_scores.append(f1_score)

        avg_f1_score = np.mean(f1_scores)
        print(f'Epoch {epoch+1} Average F1 Score: {avg_f1_score:.4f}')


class AttentionLayerTBiNet(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayerTBiNet, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # Initialize the attention weight vector with shape (features, 1)
        self.p = self.add_weight(name='attention_weight_vector',
                                 shape=(input_shape[-1], 1),
                                 initializer='random_normal',
                                 trainable=True)
        super(AttentionLayerTBiNet, self).build(input_shape)
    
    def call(self, inputs):
        # Compute attention scores using matrix multiplication, resulting in shape (batch_size, sequence_length, 1)
        attention_scores = K.dot(inputs, self.p)
        # Apply softmax over the sequence_length dimension to get attention weights
        attention_weights = K.softmax(attention_scores, axis=1)
        # Multiply inputs by attention weights (broadcasted), retaining the original input shape
        attention_output = inputs * attention_weights
        return attention_output
    
    def compute_output_shape(self, input_shape):
        # The output shape is the same as the input shape
        return input_shape