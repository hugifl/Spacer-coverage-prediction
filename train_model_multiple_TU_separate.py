#from models import CNN_BiLSTM_custom_pooling_dual_input_4_3, CNN_BiLSTM_custom_pooling_dual_input_4_2, CNN_BiLSTM_custom_pooling_dual_input_4 ,CNN_BiLSTM_custom_pooling_dual_input, CNN_BiLSTM_custom_pooling_dual_input_2, CNN_BiLSTM_avg_pooling_4_dual_input, CNN_BiLSTM_avg_pooling_4_dual_input_2
from models_2 import CNN_biLSTM_1_Masking_sep, CNN_biLSTM_Custom_Attention_Masking, CNN_biLSTM_1_Masking_TBi_Attention_sep
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from custom_elements import  poisson_loss, poisson_loss_with_masking, NaNChecker, calculate_pearson_correlation, find_and_plot_peaks, calculate_peak_f1_score, PearsonCorrelationCallback_TU, F1ScoreCallback_TU
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback
from keras.callbacks import ModelCheckpoint
from utils_training import filter_annotation_features_TU, evaluate_model, custom_batch_generator, clip_test_set, calculate_total_batches, restrict_TU_lengths, add_missing_padding, split_data_by_length,copy_model
from scipy.signal import find_peaks
import os
import re
import glob
import pandas as pd
from utils_plotting import plot_predicted_vs_observed_TU_during_training
##################### Set before training #####################

binsize = 1
batch_size = 64
dataset_name = 'Transcriptional_Units_TU_norm_V2_2'
pad_symbol = 0.420





#model_configurations = {
#    'test_pre_training': {
#        'architecture': CNN_biLSTM_1_Masking_sep,
#        'features': ['gene_vector', 'promoter_vector', 'terminator_vector', 'gene_directionality_vector'], 
#        'epochs_pre_training': 1,
#        'epochs_sep_training': 3,
#        'learning_rate': 0.005,
#        'CNN_num_layers_seq': 1,
#        'CNN_num_layers_anno': 1,
#        'filter_number_seq': [50],
#        'filter_number_anno': [50],
#        'kernel_size_seq': [3],
#        'kernel_size_anno': [3],
#        'biLSTM_num_layers_seq': 0,
#        'biLSTM_num_layers_anno': 0,
#        'unit_numbers_seq': [],
#        'unit_numbers_anno': [],
#        'unit_numbers_combined': 0,
#        'only_seq': False
#    }
#}

#model_configurations = {
#    'CNN_biLSTM_1_Masking_restr_new_sep_TBi_1': {
#        'architecture': CNN_biLSTM_1_Masking_TBi_Attention_sep,
#        'features': ['gene_vector', 'promoter_vector', 'terminator_vector', 'gene_directionality_vector'], 
#        'epochs_pre_training': 40,
#        'epochs_sep_training': 40,
#        'learning_rate': 0.005,
#        'CNN_num_layers_seq': 2,
#        'CNN_num_layers_anno': 2,
#        'filter_number_seq': [50,100],
#        'filter_number_anno': [50,100],
#        'kernel_size_seq': [3,3],
#        'kernel_size_anno': [3,3],
#        'biLSTM_num_layers_seq': 1,
#        'biLSTM_num_layers_anno': 1,
#        'unit_numbers_seq': [24],
#        'unit_numbers_anno': [24],
#        'unit_numbers_combined': 8,
#        'only_seq': False
#    }
#}

model_configurations = {
    'CNN_biLSTM_1_Masking_restr_new_sep_3': {
        'architecture': CNN_biLSTM_1_Masking_sep,
        'features': ['gene_vector', 'promoter_vector', 'terminator_vector', 'gene_directionality_vector'], 
        'epochs_pre_training': 7,
        'epochs_sep_training': 50,
        'learning_rate': 0.005,
        'CNN_num_layers_seq': 2,
        'CNN_num_layers_anno': 2,
        'filter_number_seq': [50,100],
        'filter_number_anno': [50,100],
        'kernel_size_seq': [3,3],
        'kernel_size_anno': [3,3],
        'biLSTM_num_layers_seq': 1,
        'biLSTM_num_layers_anno': 1,
        'unit_numbers_seq': [24],
        'unit_numbers_anno': [24],
        'unit_numbers_combined': 8,
        'only_seq': False
    }
}

n_split = 4
###############################################################
outdir = '../spacer_coverage_output_2/'
data_dir = '/cluster/scratch/hugifl/spacer_coverage_final_data_2/'
data_file = data_dir + dataset_name + "_data"+"/train_test_data_normalized_windows_info.npz"

data = np.load(data_file)
X_train = data['X_train']
X_test = data['X_test']
Y_train = data['Y_train']
Y_test = data['Y_test']

X_train = add_missing_padding(X_train, pad_symbol)

####### load for plotting #####################################
promoter_file = '../spacer_coverage_input/ECOCYC_promoters.txt'
terminator_file = '../spacer_coverage_input/ECOCYC_terminators.txt'
gene_file = '../spacer_coverage_input/ECOCYC_genes.txt'

promoter_df = pd.read_csv(promoter_file, sep='\t')
promoter_df.dropna(inplace=True)

terminator_df = pd.read_csv(terminator_file, sep='\t')
terminator_df.dropna(inplace=True)

gene_df = pd.read_csv(gene_file, sep='\t')
gene_df.drop(gene_df.columns[1], axis=1, inplace=True)
gene_df.dropna(inplace=True)

##############################################################


# Find rows with NaNs or Infs in Y_train
rows_with_nans_or_infs = np.any(np.isnan(Y_train) | np.isinf(Y_train), axis=1)
Y_train_filtered = Y_train[~rows_with_nans_or_infs]
X_train_filtered = X_train[~rows_with_nans_or_infs]

# Find rows with NaNs or Infs in Y_test
rows_with_nans_or_infs = np.any(np.isnan(Y_test) | np.isinf(Y_test), axis=1)
Y_test_filtered = Y_test[~rows_with_nans_or_infs]
X_test_filtered = X_test[~rows_with_nans_or_infs]

X_test_filtered, Y_test_filtered = restrict_TU_lengths(X_test_filtered, Y_test_filtered, min_length=200, max_length=3000)
X_train_filtered, Y_train_filtered = restrict_TU_lengths(X_train_filtered, Y_train_filtered, min_length=200, max_length=3000)


for model_name, config in model_configurations.items():
    checkpoint_dir = os.path.join(outdir, dataset_name + "_outputs", "checkpoints", model_name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)


    # Adjust the input data
    X_train_seq = X_train_filtered[:, :, :4]  # Sequence data
    X_train_anno = X_train_filtered[:, :, 4:] # Annotation data

    X_test_seq = X_test_filtered[:, :, :4]  # Sequence data
    X_test_anno = X_test_filtered[:, :, 4:] # Annotation data

    X_test_seq_eval, X_test_anno_eval, Y_test_filtered_eval = clip_test_set(X_test_seq, X_test_anno, Y_test_filtered)    

    # Filter the annotation arrays
    X_train_anno, X_test_anno = filter_annotation_features_TU(X_train_anno, X_test_anno, config['features'])

    ##############################################      PRE TRAINING      ########################################################
    print("Training model on full dataset")
    train_generator = custom_batch_generator(X_train_seq, X_train_anno, Y_train_filtered, batch_size)
    validation_generator = custom_batch_generator(X_test_seq, X_test_anno, Y_test_filtered, batch_size)

    train_steps_per_epoch = calculate_total_batches(Y_train_filtered, batch_size=batch_size, max_length_threshold=4000)
    val_steps_per_epoch = calculate_total_batches(Y_test_filtered, batch_size=batch_size, max_length_threshold=4000)


    early_stopping = EarlyStopping(
    monitor='val_loss',  
    min_delta=0.0005,     
    patience=5,        
    restore_best_weights=True  
    )   

    
    nan_checker = NaNChecker()
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
    pearson_callback = PearsonCorrelationCallback_TU(train_generator, num_samples=10, use_log=True, plot=False, pad_symbol= pad_symbol)
    f1_callback = F1ScoreCallback_TU(train_generator, num_samples=10, width=10, prominence=0.05, overlap_threshold=0.02, pad_symbol= pad_symbol)
    
    
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "model_epoch_{epoch:02d}"),
        save_weights_only=False, 
        save_best_only=False,
        period=5, 
        verbose=1)
    
    callbacks_list = [early_stopping, nan_checker, pearson_callback, f1_callback] #checkpoint_callback
    custom_objects = {'poisson_loss': poisson_loss_with_masking}

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
        only_seq=config['only_seq'],
        pad_symbol= pad_symbol
    )
 
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
    model.compile(optimizer=optimizer, loss=poisson_loss_with_masking, run_eagerly=True)
    
    # Filter annotation features based on model configuration
    annotation_features_to_use = config['features']
    X_train_anno_filtered, X_test_anno_filtered = filter_annotation_features_TU(X_train_anno, X_test_anno, annotation_features_to_use)

    latest_checkpoint = max(glob.glob(os.path.join(checkpoint_dir, "model_epoch_*")), key=os.path.getctime, default=None)

    epochs_already_ran = 0
    if latest_checkpoint:
        # Extract the epoch number from the checkpoint filename
        match = re.search(r"model_epoch_(\d+)", latest_checkpoint)
        if match:
            epochs_already_ran = int(match.group(1))
            print(f"Resuming from epoch {epochs_already_ran}")
            # Corrected line: assign the loaded model back to the `model` variable
            model = keras.models.load_model(latest_checkpoint, custom_objects={'poisson_loss': poisson_loss_with_masking})
        else:
            print("Could not extract epoch number from checkpoint filename, starting from scratch")
    else:
        print("No checkpoint found, starting training from scratch")

    total_epochs = config['epochs_pre_training']
    epochs_to_run = total_epochs - epochs_already_ran
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_steps_per_epoch,
        epochs=epochs_to_run,
        validation_data=validation_generator,
        validation_steps=val_steps_per_epoch,
        callbacks=callbacks_list
    )

    test_loss = model.evaluate([X_test_seq_eval, X_test_anno_eval], Y_test_filtered_eval, batch_size=len(X_test_seq_eval), verbose=0)

    # Predict on the test set
    Y_pred = model.predict([X_test_seq_eval, X_test_anno_eval])

    # Calculate average Pearson correlation and F1 score using the provided evaluate_model function
    avg_pearson_correlation, avg_f1_score = evaluate_model(Y_test_filtered_eval, Y_pred, model_name, outdir, dataset_name, pad_symbol= pad_symbol, width=10, prominence=0.05, overlap_threshold=0.02)

    # Plot the predicted vs observed coverage, loop through first dimension of Y_pred
    for i in range(Y_pred.shape[0]):
        predicted = Y_pred[i]
        observed = Y_test_filtered_eval[i]
        window_start = Y_test_filtered[i, 0]
        window_end = Y_test_filtered[i, 1]
        plot_predicted_vs_observed_TU_during_training(model_name, predicted, observed, outdir, dataset_name, promoter_df, terminator_df, gene_df, binsize, window_start, window_end, log_scale=False, pad_symbol= pad_symbol)

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

    #########################################################################  SEPERATE TRAINING ###################################################################
    print("Training model on seperate datasets")

    sub_datasets_train = split_data_by_length(X_train_seq, X_train_anno, Y_train_filtered, N=n_split)
    sub_datasets_test = split_data_by_length(X_test_seq, X_test_anno, Y_test_filtered, N=n_split)

    models = copy_model(model, N=n_split)

    for i, model in enumerate(models):
        local_model_name = f"{model_name}_split_{i}"
        X_train_seq, X_train_anno, Y_train_filtered = sub_datasets_train[i]
        X_test_seq, X_test_anno, Y_test_filtered = sub_datasets_test[i]
        X_test_seq_eval, X_test_anno_eval, Y_test_filtered_eval = clip_test_set(X_test_seq, X_test_anno, Y_test_filtered)  

        optimizer_sep = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
        model.compile(optimizer=optimizer_sep, loss=poisson_loss_with_masking, run_eagerly=True)
        
        train_generator_sep = custom_batch_generator(X_train_seq, X_train_anno, Y_train_filtered, batch_size)
        validation_generator_sep = custom_batch_generator(X_test_seq, X_test_anno, Y_test_filtered, batch_size)

        train_steps_per_epoch_sep = calculate_total_batches(Y_train_filtered, batch_size=batch_size, max_length_threshold=4000)
        val_steps_per_epoch_sep = calculate_total_batches(Y_test_filtered, batch_size=batch_size, max_length_threshold=4000)

        pearson_callback_sep = PearsonCorrelationCallback_TU(train_generator_sep, num_samples=10, use_log=True, plot=False, pad_symbol= pad_symbol)
        f1_callback_sep = F1ScoreCallback_TU(train_generator_sep, num_samples=10, width=10, prominence=0.05, overlap_threshold=0.02, pad_symbol= pad_symbol)



        callbacks_list_sep = [early_stopping, nan_checker, pearson_callback_sep, f1_callback_sep] #checkpoint_callback
        custom_objects_sep = {'poisson_loss': poisson_loss_with_masking}

        total_epochs = config['epochs_sep_training']
        epochs_to_run_sep = total_epochs
        # Train the model
        history = model.fit(
            train_generator_sep,
            steps_per_epoch=train_steps_per_epoch_sep,
            epochs=epochs_to_run_sep,
            validation_data=validation_generator_sep,
            validation_steps=val_steps_per_epoch_sep,
            callbacks=callbacks_list_sep
        )

        test_loss = model.evaluate([X_test_seq_eval, X_test_anno_eval], Y_test_filtered_eval, batch_size=len(X_test_seq_eval), verbose=0)

        # Predict on the test set
        Y_pred = model.predict([X_test_seq_eval, X_test_anno_eval])

        # Calculate average Pearson correlation and F1 score using the provided evaluate_model function
        avg_pearson_correlation, avg_f1_score = evaluate_model(Y_test_filtered_eval, Y_pred, local_model_name, outdir, dataset_name, pad_symbol= pad_symbol, width=10, prominence=0.05, overlap_threshold=0.02)

        # Plot the predicted vs observed coverage, loop through first dimension of Y_pred
        for i in range(Y_pred.shape[0]):
            predicted = Y_pred[i]
            observed = Y_test_filtered_eval[i]
            window_start = Y_test_filtered[i, 0]
            window_end = Y_test_filtered[i, 1]
            plot_predicted_vs_observed_TU_during_training(local_model_name, predicted, observed, outdir, dataset_name, promoter_df, terminator_df, gene_df, binsize, window_start, window_end, log_scale=False, pad_symbol= pad_symbol)

        # Write metrics to a text file
        metrics_filename = os.path.join(outdir, dataset_name + "_outputs", f"{local_model_name}_evaluation_metrics.txt")
        with open(metrics_filename, 'w') as file:
            file.write(f"Test Loss: {test_loss}\n")  # Assuming test_loss is a list with the loss as the first element
            file.write(f"Average Pearson Correlation: {avg_pearson_correlation:.4f}\n")
            file.write(f"Average F1 Score: {avg_f1_score:.4f}\n")

        print(f"Metrics written to {metrics_filename}")

        print(f"Average Pearson Correlation on Test Set: {avg_pearson_correlation:.4f}")
        print(f"Average F1 Score on Test Set: {avg_f1_score:.4f}")


        model.save(outdir + dataset_name + "_outputs"+"/models/" + local_model_name)

        loss_plot_directory = outdir + dataset_name + "_outputs"+f"/loss_plots_{local_model_name}/"

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
