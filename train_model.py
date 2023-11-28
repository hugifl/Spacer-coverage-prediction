from model import CNN_BiLSTM_custom_pooling_attention_poisson, CNN_BiLSTM_custom_pooling_2, MyCNNModel, MyCNNModel2, MyCNNModel_6_channel, CNN_binary, CNN_binary_dilated, CNN_binary_BiLSTM, CNN_binary_pool_first, Simple_CNN_custom_pooling_coverage
from model import CNN_BiLSTM_custom_pooling_2_poisson, CNN_custom_pooling, CNN_binary_BiLSTM_custom_pooling, CNN_binary_BiLSTM_attention_custom_pooling, CNN_binary_BiLSTM_custom_pooling_2, CNN_BiLSTM_custom_pooling_coverage
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils import NaNChecker
from custom_elements import MAE_FP_punished_more, sparse_binary_crossentropy, custom_loss_with_l1, poisson_loss
from tensorflow.keras.callbacks import EarlyStopping

# To load the correct data set
window_size = 2000
overlap = 1000

# Load your data
data = np.load('../exon_coverage_input_output/output/train_test_data_normalized_'+str(window_size) + '_' + str(overlap) + '.npz')
X_train = data['X_train']
X_test = data['X_test']
Y_train = data['Y_train']
Y_test = data['Y_test']

# Filter out windows that contain genes with coverage peaks too high (normalization error due to wrong/non-matching coordinates) or too low (low gene expression, noisy profile)

indices_to_remove_train = np.where((Y_train > 20).any(axis=1) | (Y_train.max(axis=1) < 2))[0]

# Remove these rows from Y_train and X_train
Y_train_filtered = np.delete(Y_train, indices_to_remove_train, axis=0)
X_train_filtered = np.delete(X_train, indices_to_remove_train, axis=0)

# Find indices where the maximum value in a row of Y_test exceeds 30 or is below 2
indices_to_remove_test = np.where((Y_test > 20).any(axis=1) | (Y_test.max(axis=1) < 2))[0]

# Remove these rows from Y_test and X_test
Y_test_filtered = np.delete(Y_test, indices_to_remove_test, axis=0)
X_test_filtered = np.delete(X_test, indices_to_remove_test, axis=0)

## Find rows with NaNs or Infs in Y_train
#rows_with_nans_or_infs = np.any(np.isnan(Y_train) | np.isinf(Y_train), axis=1)
#Y_train_filtered = Y_train[~rows_with_nans_or_infs]
#X_train_filtered = X_train[~rows_with_nans_or_infs]
#
## Find rows with NaNs or Infs in Y_test
#rows_with_nans_or_infs = np.any(np.isnan(Y_test) | np.isinf(Y_test), axis=1)
#Y_test_filtered = Y_test[~rows_with_nans_or_infs]
#X_test_filtered = X_test[~rows_with_nans_or_infs]

small_constant = 1e-6
Y_train_log = np.log10(Y_train_filtered + small_constant)
Y_test_log = np.log10(Y_test_filtered + small_constant)

# Remove channels 
#X_train_filtered = X_train_filtered[:, :, :-2]  
#X_test_filtered = X_test_filtered[:, :, :-2]  

early_stopping = EarlyStopping(
    monitor='val_loss',  
    min_delta=0.001,     
    patience=10,        
    restore_best_weights=True  
)

nan_checker = NaNChecker()
model = CNN_BiLSTM_custom_pooling_2_poisson()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

model.compile(optimizer=optimizer, loss = poisson_loss, metrics=['accuracy'])  # custom_loss_with_l1 weighted_binary_crossentropy tf.keras.losses.Poisson() 'mean_squared_error' MeanAbsoluteError() MAE_FP_punished_more sparse_binary_crossentropy



# Train the model
history = model.fit(X_train_filtered, Y_train_filtered, epochs=450, batch_size=32, validation_data=(X_test_filtered, Y_test_filtered), callbacks=[early_stopping,nan_checker])

# Evaluate the model
model.evaluate(X_test_filtered, Y_test_filtered)
model.save('../exon_coverage_input_output/output/'+str(window_size) + '_' + str(overlap) + 'CNN_BiLSTM_custom_pooling_2_poisson_4')

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