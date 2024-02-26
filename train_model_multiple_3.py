#from models import CNN_BiLSTM_custom_pooling_dual_input_4_3, CNN_BiLSTM_custom_pooling_dual_input_4_2, CNN_BiLSTM_custom_pooling_dual_input_4 ,CNN_BiLSTM_custom_pooling_dual_input, CNN_BiLSTM_custom_pooling_dual_input_2, CNN_BiLSTM_avg_pooling_4_dual_input, CNN_BiLSTM_avg_pooling_4_dual_input_2
from models_2 import CNN, CNN_binned, biLSTM, CNN_BiLSTM_custom_pooling_dual_input_old, CNN_biLSTM_1, CNN_biLSTM_AddAttention, CNN_biLSTM_CustomAttention, CNN_biLSTM_CustomAttention_2
import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from custom_elements import  poisson_loss, NaNChecker, calculate_pearson_correlation, find_and_plot_peaks, calculate_peak_f1_score, PearsonCorrelationCallback, F1ScoreCallback
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback
from keras.callbacks import ModelCheckpoint
from utils_training import filter_annotation_features, evaluate_model
from scipy.signal import find_peaks
import os
import re
import glob
##################### Set before training #####################

window_size = 3200
overlap = 1600
no_bin = 3200
binsize = 1
batch_size = 64
dataset_name = '3200_1600_gene_norm'


model_configurations = {
    'CNN_biLSTM_CustomAttention_1_4': {
        'architecture': CNN_biLSTM_CustomAttention,
        'features': ['gene_vector', 'promoter_vector', 'terminator_vector', 'gene_directionality_vector'],
        'epochs': 50,
        'learning_rate': 0.005,
        'CNN_num_layers_seq': 2,
        'CNN_num_layers_anno': 2,
        'filter_number_seq': [100,100],
        'filter_number_anno': [100,100],
        'kernel_size_seq': [5,5],
        'kernel_size_anno': [5,5],
        'biLSTM_num_layers_seq': 0,
        'biLSTM_num_layers_anno': 0,
        'unit_numbers_seq': [],
        'unit_numbers_anno': [],
        'unit_numbers_combined': 64,
        'only_seq': False
    }
}
#model_configurations = {
#    'CNN_biLSTM_13': {
#        'architecture': CNN_biLSTM_1,
#        'features': ['gene_vector', 'promoter_vector', 'terminator_vector', 'gene_directionality_vector'],
#        'epochs': 50,
#        'learning_rate': 0.005,
#        'CNN_num_layers_seq': 2,
#        'CNN_num_layers_anno': 2,
#        'filter_number_seq': [100,100],
#        'filter_number_anno': [100,100],
#        'kernel_size_seq': [5,5],
#        'kernel_size_anno': [5,5],
#        'biLSTM_num_layers_seq': 0,
#        'biLSTM_num_layers_anno': 0,
#        'unit_numbers_seq': [],
#        'unit_numbers_anno': [],
#        'unit_numbers_combined': 8,
#        'only_seq': False
#    },
#    'CNN_biLSTM_AddAttention_1': {
#        'architecture': CNN_biLSTM_AddAttention,
#        'features': ['gene_vector', 'promoter_vector', 'terminator_vector', 'gene_directionality_vector'],
#        'epochs': 50,
#        'learning_rate': 0.005,
#        'CNN_num_layers_seq': 2,
#        'CNN_num_layers_anno': 2,
#        'filter_number_seq': [100,100],
#        'filter_number_anno': [100,100],
#        'kernel_size_seq': [5,5],
#        'kernel_size_anno': [5,5],
#        'biLSTM_num_layers_seq': 0,
#        'biLSTM_num_layers_anno': 0,
#        'unit_numbers_seq': [],
#        'unit_numbers_anno': [],
#        'unit_numbers_combined': 8,
#        'only_seq': False
#    }
#}

#model_configurations = {
#    
#    'biLSTM_4': {
#        'architecture': biLSTM,
#        'features': ['gene_vector', 'promoter_vector', 'terminator_vector', 'gene_directionality_vector'],
#        'epochs': 50,
#        'learning_rate': 0.005,
#        'num_layers_seq': 1,
#        'num_layers_anno': 1,
#        'unit_numbers_seq': [32],
#        'unit_numbers_anno': [32],
#        'only_seq': False
#    },
#    
#    'biLSTM_1': {
#        'architecture': biLSTM,
#        'features': ['gene_vector', 'promoter_vector', 'terminator_vector', 'gene_directionality_vector'],
#        'epochs': 50,
#        'learning_rate': 0.005,
#        'num_layers_seq': 1,
#        'num_layers_anno': 1,
#        'unit_numbers_seq': [32],
#        'unit_numbers_anno': [32],
#        'only_seq': True
#    },
#    
#    'biLSTM_5': {
#        'architecture': biLSTM,
#        'features': ['gene_vector', 'promoter_vector', 'terminator_vector', 'gene_directionality_vector'],
#        'epochs': 50,
#        'learning_rate': 0.005,
#        'num_layers_seq': 2,
#        'num_layers_anno': 2,
#        'unit_numbers_seq': [32,32],
#        'unit_numbers_anno': [32,32],
#        'only_seq': False
#    },
#    'biLSTM_2': {
#        'architecture': biLSTM,
#        'features': ['gene_vector', 'promoter_vector', 'terminator_vector', 'gene_directionality_vector'],
#        'epochs': 50,
#        'learning_rate': 0.005,
#        'num_layers_seq': 2,
#        'num_layers_anno': 2,
#        'unit_numbers_seq': [32,32],
#        'unit_numbers_anno': [32,32],
#        'only_seq': True
#    }
#}

#model_configurations = {
#    
#    'tetstets': {
#        'architecture': CNN,
#        'features': ['gene_vector', 'promoter_vector', 'terminator_vector', 'gene_directionality_vector'],
#        'epochs': 16,
#        'learning_rate': 0.001,
#        'num_layers_seq': 1,
#        'num_layers_anno': 1,
#        'filter_number_seq': [50],
#        'filter_number_anno': [50],
#        'kernel_size_seq': [3],
#        'kernel_size_anno': [3],
#        'only_seq': True
#    }
#}

coverage_scaling_factor = 1
###############################################################
outdir = '../spacer_coverage_output_2/'
data_dir = '/cluster/scratch/hugifl/spacer_coverage_final_data_2/'
data_file = data_dir + dataset_name + "_data"+"/train_test_data_normalized_windows_info.npz"

data = np.load(data_file)
X_train = data['X_train']
X_test = data['X_test']
Y_train = data['Y_train']
Y_test = data['Y_test']

# Adjust the coverage data
Y_train = Y_train[:, 2:]
Y_test = Y_test[:, 2:]

Y_test = Y_test * coverage_scaling_factor
Y_train = Y_train * coverage_scaling_factor


# Find rows with NaNs or Infs in Y_train
rows_with_nans_or_infs = np.any(np.isnan(Y_train) | np.isinf(Y_train), axis=1)
Y_train_filtered = Y_train[~rows_with_nans_or_infs]
X_train_filtered = X_train[~rows_with_nans_or_infs]

# Find rows with NaNs or Infs in Y_test
rows_with_nans_or_infs = np.any(np.isnan(Y_test) | np.isinf(Y_test), axis=1)
Y_test_filtered = Y_test[~rows_with_nans_or_infs]
X_test_filtered = X_test[~rows_with_nans_or_infs]


# Filter out windows that contain genes with coverage peaks too high (normalization error due to wrong/non-matching coordinates) or too low (low gene expression, noisy profile)
#indices_to_remove_train = np.where((Y_train_filtered > 200).any(axis=1) | (Y_train_filtered.max(axis=1) < 5))[0]
##
### Remove these rows from Y_train and X_train
#Y_train_filtered = np.delete(Y_train_filtered, indices_to_remove_train, axis=0)
#X_train_filtered = np.delete(X_train_filtered, indices_to_remove_train, axis=0)
#
## Find indices where the maximum value in a row of Y_test exceeds 20 or is below 2
#indices_to_remove_test = np.where((Y_test_filtered > 200).any(axis=1) | (Y_test_filtered.max(axis=1) < 5))[0]
##
### Remove these rows from Y_test and X_test
#Y_test_filtered = np.delete(Y_test_filtered, indices_to_remove_test, axis=0)
#X_test_filtered = np.delete(X_test_filtered, indices_to_remove_test, axis=0)

#Y_train_binarized = (Y_train_filtered > 2).astype(int)
#Y_test_binarized = (Y_test_filtered > 2).astype(int)

for model_name, config in model_configurations.items():
    
    checkpoint_dir = os.path.join(outdir, dataset_name + "_outputs", "checkpoints", model_name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)


    # Adjust the input data
    X_train_seq = X_train_filtered[:, :, :4]  # Sequence data
    X_train_anno = X_train_filtered[:, :, 4:] # Annotation data

    X_test_seq = X_test_filtered[:, :, :4]  # Sequence data
    X_test_anno = X_test_filtered[:, :, 4:] # Annotation data

    # Filter the annotation arrays
    X_train_anno, X_test_anno = filter_annotation_features(X_train_anno, X_test_anno, config['features'])
    
    early_stopping = EarlyStopping(
    monitor='val_loss',  
    min_delta=0.0005,     
    patience=5,        
    restore_best_weights=True  
    )   

    
    nan_checker = NaNChecker()
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
    pearson_callback = PearsonCorrelationCallback(X_train_seq, X_train_anno, Y_train_filtered, batch_size=batch_size, use_log=True, plot=False)
    f1_callback = F1ScoreCallback(X_train_seq, X_train_anno, Y_train_filtered, batch_size=batch_size, width=10, prominence=0.05, overlap_threshold=0.02, data_length=no_bin)
    
    
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "model_epoch_{epoch:02d}"),
        save_weights_only=False, 
        save_best_only=False,
        period=5, 
        verbose=1)
    
    callbacks_list = [early_stopping, nan_checker, pearson_callback, f1_callback, checkpoint_callback]
    custom_objects = {'poisson_loss': poisson_loss}

    # Setup model based on configuration
    model = config['architecture'](
        CNN_num_layers_seq=config['CNN_num_layers_seq'],
        CNN_num_layers_anno=config['CNN_num_layers_anno'],
        filter_number_seq=config['filter_number_seq'],
        filter_number_anno=config['filter_number_anno'],
        kernel_size_seq=config['kernel_size_seq'],
        kernel_size_anno=config['kernel_size_anno'],
        biLSTM_num_layers_seq=config['biLSTM_num_layers_seq'],
        biLSTM_num_layers_anno=config['biLSTM_num_layers_anno'],
        unit_numbers_seq=config['unit_numbers_seq'],
        unit_numbers_anno=config['unit_numbers_anno'],
        unit_numbers_combined=config['unit_numbers_combined'],
        only_seq=config['only_seq']
    )
 
    #model = config['architecture'](
    #    num_layers_seq=config['num_layers_seq'],
    #    num_layers_anno=config['num_layers_anno'],
    #    unit_numbers_seq=config['unit_numbers_seq'],
    #    unit_numbers_anno=config['unit_numbers_anno'],
    #    only_seq=config['only_seq']
    #)
    #model = config['architecture'](
    #    num_layers_seq=config['num_layers_seq'],
    #    num_layers_anno=config['num_layers_anno'],
    #    filter_number_seq=config['filter_number_seq'],
    #    filter_number_anno=config['filter_number_anno'],
    #    kernel_size_seq=config['kernel_size_seq'],
    #    kernel_size_anno=config['kernel_size_anno'],
    #    only_seq=config['only_seq']
    #)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
    model.compile(optimizer=optimizer, loss=poisson_loss, run_eagerly=True)
    
    # Filter annotation features based on model configuration
    annotation_features_to_use = config['features']
    X_train_anno_filtered, X_test_anno_filtered = filter_annotation_features(X_train_anno, X_test_anno, annotation_features_to_use)

    latest_checkpoint = max(glob.glob(os.path.join(checkpoint_dir, "model_epoch_*")), key=os.path.getctime, default=None)

    epochs_already_ran = 0
    if latest_checkpoint:
        # Extract the epoch number from the checkpoint filename
        match = re.search(r"model_epoch_(\d+)", latest_checkpoint)
        if match:
            epochs_already_ran = int(match.group(1))
            print(f"Resuming from epoch {epochs_already_ran}")
            # Corrected line: assign the loaded model back to the `model` variable
            model = keras.models.load_model(latest_checkpoint, custom_objects={'poisson_loss': poisson_loss})
        else:
            print("Could not extract epoch number from checkpoint filename, starting from scratch")
    else:
        print("No checkpoint found, starting training from scratch")

    total_epochs = config['epochs']
    epochs_to_run = total_epochs - epochs_already_ran
    # Train the model
    history = model.fit(
        [X_train_seq, X_train_anno_filtered],
        Y_train_filtered,
        epochs=epochs_to_run,
        batch_size=batch_size,
        validation_data=([X_test_seq, X_test_anno_filtered], Y_test_filtered),
        callbacks=callbacks_list
    )

    test_loss = model.evaluate([X_test_seq, X_test_anno], Y_test, verbose=0)

    # Predict on the test set
    Y_pred = model.predict([X_test_seq, X_test_anno])

    # Calculate average Pearson correlation and F1 score using the provided evaluate_model function
    avg_pearson_correlation, avg_f1_score = evaluate_model(Y_test, Y_pred, model_name, outdir, dataset_name, width=10, prominence=0.05, overlap_threshold=0.02)

    # Write metrics to a text file
    metrics_filename = os.path.join(outdir, dataset_name + "_outputs", f"{model_name}_evaluation_metrics.txt")
    with open(metrics_filename, 'w') as file:
        file.write(f"Test Loss: {test_loss}\n")  # Assuming test_loss is a list with the loss as the first element
        file.write(f"Average Pearson Correlation: {avg_pearson_correlation:.4f}\n")
        file.write(f"Average F1 Score: {avg_f1_score:.4f}\n")

    print(f"Metrics written to {metrics_filename}")

    print(f"Average Pearson Correlation on Test Set: {avg_pearson_correlation:.4f}")
    print(f"Average F1 Score on Test Set: {avg_f1_score:.4f}")


    model.save(outdir + dataset_name + "_outputs"+"/models/" + model_name)

    loss_plot_directory = outdir + dataset_name + "_outputs"+f"/loss_plots_{model_name}/"

    if not os.path.exists(loss_plot_directory):
        os.makedirs(loss_plot_directory)

    # Plot training & validation loss values
    plt.style.use('ggplot')
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training loss')
    plt.title('Model Loss Over Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch') 
    plt.legend()
    plt.savefig(loss_plot_directory + "training_loss.png")
    plt.close()

    plt.style.use('ggplot')
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Model Loss Over Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(loss_plot_directory + "validation_loss.png")
    plt.close()