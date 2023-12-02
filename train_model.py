from models import CNN_BiLSTM_custom_pooling_attention_poisson, CNN_BiLSTM_custom_pooling_2, Simple_CNN_custom_pooling_coverage
from models import CNN_binary_BiLSTM_custom_pooling_3, CNN_BiLSTM_custom_pooling_poisson_bin32, CNN_BiLSTM_custom_pooling_poisson_bin2, CNN_BiLSTM_custom_pooling_2_poisson, CNN_custom_pooling, CNN_binary_BiLSTM_custom_pooling, CNN_binary_BiLSTM_attention_custom_pooling, CNN_binary_BiLSTM_custom_pooling_2, CNN_BiLSTM_custom_pooling_coverage
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils import NaNChecker
from custom_elements import custom_loss_with_l1, poisson_loss
from tensorflow.keras.callbacks import EarlyStopping

# To load the correct data set
window_size = 3200
overlap = 1602

# Load your data
data = np.load('../exon_coverage_input_output/output/train_test_data_normalized_windows_info_'+str(window_size) + '_' + str(overlap) + '.npz')
X_train = data['X_train']
X_test = data['X_test']
Y_train = data['Y_train']
Y_test = data['Y_test']

# Remove the channels containing the window information
Y_train = Y_train[:, 2:]
Y_test = Y_test[:, 2:]

scaling_factor = 1e-6

Y_test = Y_test * scaling_factor
Y_train = Y_train * scaling_factor
# Filter out windows that contain genes with coverage peaks too high (normalization error due to wrong/non-matching coordinates) or too low (low gene expression, noisy profile)

indices_to_remove_train = np.where((Y_train > 15).any(axis=1) | (Y_train.max(axis=1) < 2))[0]
#
## Remove these rows from Y_train and X_train
Y_train_filtered = np.delete(Y_train, indices_to_remove_train, axis=0)
X_train_filtered = np.delete(X_train, indices_to_remove_train, axis=0)
#
## Find indices where the maximum value in a row of Y_test exceeds 30 or is below 2
indices_to_remove_test = np.where((Y_test > 15).any(axis=1) | (Y_test.max(axis=1) < 2))[0]
#
## Remove these rows from Y_test and X_test
Y_test_filtered = np.delete(Y_test, indices_to_remove_test, axis=0)
X_test_filtered = np.delete(X_test, indices_to_remove_test, axis=0)

Y_train_binarized = (Y_train_filtered > 2).astype(int)
Y_test_binarized = (Y_test_filtered > 2).astype(int)

## Find rows with NaNs or Infs in Y_train
#rows_with_nans_or_infs = np.any(np.isnan(Y_train) | np.isinf(Y_train), axis=1)
#Y_train_filtered = Y_train[~rows_with_nans_or_infs]
#X_train_filtered = X_train[~rows_with_nans_or_infs]
#
## Find rows with NaNs or Infs in Y_test
#rows_with_nans_or_infs = np.any(np.isnan(Y_test) | np.isinf(Y_test), axis=1)
#Y_test_filtered = Y_test[~rows_with_nans_or_infs]
#X_test_filtered = X_test[~rows_with_nans_or_infs]

#small_constant = 1e-6
#Y_train_log = np.log10(Y_train_filtered + small_constant)
#Y_test_log = np.log10(Y_test_filtered + small_constant)

# Remove channels 
#X_train_filtered = X_train_filtered[:, :, :-2]  
#X_test_filtered = X_test_filtered[:, :, :-2]  

early_stopping = EarlyStopping(
    monitor='val_loss',  
    min_delta=0.0005,     
    patience=15,        
    restore_best_weights=True  
)

nan_checker = NaNChecker()
model = CNN_binary_BiLSTM_custom_pooling_3()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss = 'binary_crossentropy', metrics=['accuracy'])  # custom_loss_with_l1 weighted_binary_crossentropy tf.keras.losses.Poisson() 'mean_squared_error' MeanAbsoluteError() MAE_FP_punished_more sparse_binary_crossentropy



# Train the model
history = model.fit(X_train_filtered, Y_train_binarized, epochs=6, batch_size=32, validation_data=(X_test_filtered, Y_test_binarized), callbacks=[early_stopping,nan_checker])

# Evaluate the model
model.evaluate(X_test_filtered, Y_test_binarized)
model.save('../exon_coverage_input_output/output/models_'+str(window_size) + '_' + str(overlap)+ '_bin_2_unnormalized'+'/' +str(window_size) + '_' + str(overlap) + 'CNN_BiLSTM_custom_pooling_binary')

# Plot training & validation loss values
plt.style.use('ggplot')
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training loss')
plt.title('Model Loss Over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch') 
plt.legend()
plt.savefig("../exon_coverage_input_output/output/training_loss.png")
plt.close()

plt.style.use('ggplot')
plt.figure(figsize=(12, 6))
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Model Loss Over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig("../exon_coverage_input_output/output/validation_loss.png")
plt.close()