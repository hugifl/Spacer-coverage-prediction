from models import CNN_BiLSTM_custom_pooling_dual_input_4_3, CNN_BiLSTM_custom_pooling_dual_input_4_2, CNN_BiLSTM_custom_pooling_dual_input_4 ,CNN_BiLSTM_custom_pooling_dual_input, CNN_BiLSTM_custom_pooling_dual_input_2, CNN_BiLSTM_avg_pooling_4_dual_input, CNN_BiLSTM_avg_pooling_4_dual_input_2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from custom_elements import  poisson_loss, NaNChecker, spearman_correlation
from tensorflow.keras.callbacks import EarlyStopping

##################### Set before training #####################

window_size = 3200
overlap = 1600
no_bin = 3200
binsize = 1
dataset_name = 'paraquat_window_3200_overlapt_1600_binsize_4'
model_name = 'CNN_BiLSTM_custom_pooling_dual_input_4_2'
model = CNN_BiLSTM_custom_pooling_dual_input_4_2()

learning_rate = 0.0005
erly_stopping_patience = 10
epochs = 200
###############################################################

outdir = '../spacer_coverage_output_2/'
data_dir = '/cluster/scratch/hugifl/spacer_coverage_final_data/'
data_file = data_dir + dataset_name + "_data"+"/train_test_data_normalized_windows_info_smoothed.npz"
data = np.load(data_file)
X_train = data['X_train']
X_test = data['X_test']
Y_train = data['Y_train']
Y_test = data['Y_test']
# Adjust the coverage data
Y_train = Y_train[:, 2:]
Y_test = Y_test[:, 2:]

scaling_factor = 1

Y_test = Y_test * scaling_factor
Y_train = Y_train * scaling_factor


# Find rows with NaNs or Infs in Y_train
rows_with_nans_or_infs = np.any(np.isnan(Y_train) | np.isinf(Y_train), axis=1)
Y_train_filtered = Y_train[~rows_with_nans_or_infs]
X_train_filtered = X_train[~rows_with_nans_or_infs]

# Find rows with NaNs or Infs in Y_test
rows_with_nans_or_infs = np.any(np.isnan(Y_test) | np.isinf(Y_test), axis=1)
Y_test_filtered = Y_test[~rows_with_nans_or_infs]
X_test_filtered = X_test[~rows_with_nans_or_infs]


# Filter out windows that contain genes with coverage peaks too high (normalization error due to wrong/non-matching coordinates) or too low (low gene expression, noisy profile)
indices_to_remove_train = np.where((Y_train_filtered > 200).any(axis=1) | (Y_train_filtered.max(axis=1) < 5))[0]
#
## Remove these rows from Y_train and X_train
Y_train_filtered = np.delete(Y_train_filtered, indices_to_remove_train, axis=0)
X_train_filtered = np.delete(X_train_filtered, indices_to_remove_train, axis=0)
#
## Find indices where the maximum value in a row of Y_test exceeds 20 or is below 2
indices_to_remove_test = np.where((Y_test_filtered > 200).any(axis=1) | (Y_test_filtered.max(axis=1) < 5))[0]
#
## Remove these rows from Y_test and X_test
Y_test_filtered = np.delete(Y_test_filtered, indices_to_remove_test, axis=0)
X_test_filtered = np.delete(X_test_filtered, indices_to_remove_test, axis=0)

#Y_train_binarized = (Y_train_filtered > 2).astype(int)
#Y_test_binarized = (Y_test_filtered > 2).astype(int)

# Adjust the input data
X_train_seq = X_train_filtered[:, :, :4]  # Sequence data
X_train_anno = X_train_filtered[:, :, 4:] # Annotation data

X_test_seq = X_test_filtered[:, :, :4]  # Sequence data
X_test_anno = X_test_filtered[:, :, 4:] # Annotation data



early_stopping = EarlyStopping(
    monitor='val_loss',  
    min_delta=0.0005,     
    patience=erly_stopping_patience,        
    restore_best_weights=True  
)

nan_checker = NaNChecker()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer, loss = poisson_loss, metrics=[spearman_correlation], run_eagerly=True)  # custom_loss_with_l1 weighted_binary_crossentropy tf.keras.losses.Poisson() 'mean_squared_error' MeanAbsoluteError() MAE_FP_punished_more sparse_binary_crossentropy

#model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)


# Train the model
history = model.fit(
    [X_train_seq, X_train_anno], 
    Y_train_filtered, 
    epochs=epochs, 
    batch_size=32, 
    validation_data=([X_test_seq, X_test_anno], Y_test_filtered), 
    callbacks=[early_stopping, nan_checker]
)
# Evaluate the model
model.evaluate([X_test_seq, X_test_anno], Y_test_filtered)
model.save(outdir + dataset_name + "_outputs"+"/models/" + model_name)

# Plot training & validation loss values
plt.style.use('ggplot')
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training loss')
plt.title('Model Loss Over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch') 
plt.legend()
plt.savefig(outdir + dataset_name + "_outputs"+"/loss_plots/" + model_name + "training_loss.png")
plt.close()

plt.style.use('ggplot')
plt.figure(figsize=(12, 6))
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Model Loss Over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig(outdir + dataset_name + "_outputs"+"/loss_plots/" + model_name + "validation_loss.png")
plt.close()