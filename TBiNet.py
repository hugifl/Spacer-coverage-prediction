from keras.models import load_model

model_file_path = '../exon_coverage_input_output/tbinet_best.hdf5'
print("test")


# Load the pre-trained model
model = load_model(model_file_path)

# Print the model's architecture
model.summary()
