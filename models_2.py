import tensorflow as tf
from custom_elements import CustomPooling, AttentionMechanism
from keras.initializers import HeNormal
from tensorflow.keras.layers import Conv1D, ELU, BatchNormalization, AveragePooling1D
from tensorflow.keras import backend as K
from tensorflow.keras.layers import concatenate

Model = tf.keras.models.Model
Conv1D = tf.keras.layers.Conv1D
MaxPooling1D = tf.keras.layers.MaxPooling1D
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense
Input = tf.keras.layers.Input
Dropout = tf.keras.layers.Dropout
Bidirectional = tf.keras.layers.Bidirectional
LSTM = tf.keras.layers.LSTM

class CNN(Model):
    def __init__(self, num_layers_seq, num_layers_anno, filter_number_seq, filter_number_anno, kernel_size_seq, kernel_size_anno, only_seq):
        super(CNN, self).__init__()

        # Assertions for hyperparameter list lengths
        assert len(filter_number_seq) >= num_layers_seq and len(kernel_size_seq) >= num_layers_seq, "Sequence stream lists must match the number of layers"
        assert len(filter_number_anno) >= num_layers_anno and len(kernel_size_anno) >= num_layers_anno, "Annotation stream lists must match the number of layers"

        # DNA sequence stream layers
        self.seq_layers = [Conv1D(filters=filter_number_seq[i], kernel_size=kernel_size_seq[i], padding='same', activation='relu') for i in range(num_layers_seq)]
        self.drop_seq = Dropout(0.2)

        self.only_seq = only_seq  

        if not only_seq:
            # Gene/operon annotation stream layers
            self.anno_layers = [Conv1D(filters=filter_number_anno[i], kernel_size=kernel_size_anno[i], padding='same', activation='relu') for i in range(num_layers_anno)]
            self.drop_anno = Dropout(0.2)

        # Shared layers
        self.final_conv = Conv1D(filters=1, kernel_size=1, activation='relu', padding='same')

    def call(self, inputs):
        inputs_seq, inputs_anno = inputs

        # DNA sequence stream
        x_seq = inputs_seq
        for layer in self.seq_layers:
            x_seq = layer(x_seq)
        x_seq = self.drop_seq(x_seq)

        if not self.only_seq:
            # Gene/operon annotation stream
            x_anno = inputs_anno
            for layer in self.anno_layers:
                x_anno = layer(x_anno)
            x_anno = self.drop_anno(x_anno)
            combined = concatenate([x_seq, x_anno])
        else:
            combined = x_seq

        # Shared layers
        return self.final_conv(combined)


class CNN_1_layer(Model): # better than previous
    def __init__(self):
        super(CNN_1_layer, self).__init__()
        # DNA sequence stream
        self.conv1_seq = Conv1D(filters=100, kernel_size=5, padding='same')
        self.drop2_seq = Dropout(0.2)

        # Gene/operon annotation stream
        self.conv1_anno = Conv1D(filters=100, kernel_size=5, padding='same')
        self.drop2_anno = Dropout(0.2)

        # Shared layers
        #self.bilstm = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))
        #self.drop3 = Dropout(0.2)
        self.final_conv = Conv1D(filters=1, kernel_size=1, activation='relu', padding='same')

    def call(self, inputs):
        inputs_seq, inputs_anno = inputs
        # DNA sequence stream
        x_seq = self.conv1_seq(inputs_seq)
        x_seq = self.drop2_seq(x_seq)

        # Gene/operon annotation stream
        x_anno = self.conv1_anno(inputs_anno)
        x_anno = self.batchnorm1_anno(x_anno, training=False)
        x_anno = self.drop2_anno(x_anno)

        # Combine the outputs of both streams
        combined = concatenate([x_seq, x_anno])

        # Shared layers
        return self.final_conv(combined)
    

    
############################################### COVERAGE AS PROBABILITY APPROACH ###############################################
    


class CNN_BiLSTM_two_outputs_1(Model): # better than previous
    def __init__(self):
        super(CNN_BiLSTM_two_outputs_1, self).__init__()
        # DNA sequence stream
        self.conv1_seq = Conv1D(filters=100, kernel_size=4, padding='same')
        self.batchnorm1_seq = BatchNormalization()  # Batch normalization layer
        self.pool1_seq = CustomPooling(pool_size=2, strides=2)
        self.pool2_seq = CustomPooling(pool_size=2, strides=2)
        self.conv3_seq = Conv1D(filters=150, kernel_size=8, padding='same')
        self.batchnorm3_seq = BatchNormalization()  # Batch normalization layer
        self.drop2_seq = Dropout(0.2)

        # Gene/operon annotation stream
        self.conv1_anno = Conv1D(filters=50, kernel_size=4, padding='same')
        self.batchnorm1_anno = BatchNormalization()  # Batch normalization layer
        self.pool1_anno = CustomPooling(pool_size=2, strides=2)
        self.pool2_anno = CustomPooling(pool_size=2, strides=2)
        self.conv2_anno = Conv1D(filters=100, kernel_size=8, padding='same')
        self.batchnorm2_anno = BatchNormalization()  # Batch normalization layer
        self.drop2_anno = Dropout(0.2)

        # Shared layers
        self.bilstm = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))
        self.drop3 = Dropout(0.2)
        self.final_conv = Conv1D(filters=2, kernel_size=1, activation='softmax', padding='same')

    def call(self, inputs):
        inputs_seq, inputs_anno = inputs
        # DNA sequence stream
        x_seq = self.conv1_seq(inputs_seq)
        x_seq = self.batchnorm1_seq(x_seq, training=False)
        x_seq = self.pool1_seq(x_seq)
        x_seq = self.pool2_seq(x_seq)
        x_seq = self.conv3_seq(x_seq)
        x_seq = self.batchnorm3_seq(x_seq, training=False)
        x_seq = self.drop2_seq(x_seq)

        # Gene/operon annotation stream
        x_anno = self.conv1_anno(inputs_anno)
        x_anno = self.batchnorm1_anno(x_anno, training=False)
        x_anno = self.pool1_anno(x_anno)
        x_anno = self.pool2_anno(x_anno)
        x_anno = self.conv2_anno(x_anno)
        x_anno = self.batchnorm2_anno(x_anno, training=False)
        x_anno = self.drop2_anno(x_anno)

        # Combine the outputs of both streams
        combined = concatenate([x_seq, x_anno])

        # Shared layers
        x = self.bilstm(combined)
        x = self.drop3(x)
        return self.final_conv(x)