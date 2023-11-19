import h5py
import tensorflow as tf

model_file_path = 'exon_coverage_input_output/tbinet_best.hdf5'
print("test")

model_file = h5py.File(model_file_path, 'r')

# Access the model's architecture and weights. The 'model' group is commonly used to store both.
model = model_file['model']
