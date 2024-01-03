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



class CNN_BiLSTM_custom_pooling_poisson_bin2(Model):
    def __init__(self):
        super(CNN_BiLSTM_custom_pooling_poisson_bin2, self).__init__()
        self.conv1 = Conv1D(filters=100, kernel_size=6, activation='relu', padding='same') # 100 4
        self.conv3 = Conv1D(filters=150, kernel_size=8, activation='relu', padding='same')
        #self.drop1 = Dropout(0.2)
        self.pool1 = CustomPooling(pool_size=2, strides=2)
        self.conv2 = Conv1D(filters=200, kernel_size=4, activation='relu', padding='same') #  100 4
        self.drop2 = Dropout(0.2)
        self.bilstm = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))
        self.drop3 = Dropout(0.2)
        self.final_conv = Conv1D(filters=1, kernel_size=1, activation='relu', padding='same')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv3(x)
        #x = self.drop1(x)   
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.drop2(x)  
        x = self.bilstm(x)
        x = self.drop3(x)
        return self.final_conv(x)


class CNN_BiLSTM_custom_pooling_dual_input(Model):
    def __init__(self):
        super(CNN_BiLSTM_custom_pooling_dual_input, self).__init__()
        # DNA sequence stream
        self.conv1_seq = Conv1D(filters=100, kernel_size=6, activation='relu', padding='same')
        self.conv3_seq = Conv1D(filters=150, kernel_size=8, activation='relu', padding='same')
        self.pool1_seq = CustomPooling(pool_size=2, strides=2)
        #self.conv2_seq = Conv1D(filters=200, kernel_size=4, activation='relu', padding='same')
        self.drop2_seq = Dropout(0.2)

        # Gene/operon annotation stream
        self.conv1_anno = Conv1D(filters=100, kernel_size=6, activation='relu', padding='same')
        #self.conv3_anno = Conv1D(filters=150, kernel_size=8, activation='relu', padding='same')
        self.pool1_anno = CustomPooling(pool_size=2, strides=2)
        #self.conv2_anno = Conv1D(filters=200, kernel_size=4, activation='relu', padding='same')
        self.drop2_anno = Dropout(0.2)

        # Shared layers
        self.bilstm = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))
        self.drop3 = Dropout(0.2)
        self.final_conv = Conv1D(filters=1, kernel_size=1, activation='relu', padding='same')

    def call(self, inputs):
        inputs_seq, inputs_anno = inputs
        # DNA sequence stream
        x_seq = self.conv1_seq(inputs_seq)
        x_seq = self.conv3_seq(x_seq)
        x_seq = self.pool1_seq(x_seq)
        #x_seq = self.conv2_seq(x_seq)
        x_seq = self.drop2_seq(x_seq)

        # Gene/operon annotation stream
        x_anno = self.conv1_anno(inputs_anno)
        #x_anno = self.conv3_anno(x_anno)
        x_anno = self.pool1_anno(x_anno)
        #x_anno = self.conv2_anno(x_anno)
        x_anno = self.drop2_anno(x_anno)

        # Combine the outputs of both streams
        combined = concatenate([x_seq, x_anno])

        # Shared layers
        x = self.bilstm(combined)
        x = self.drop3(x)
        return self.final_conv(x)
    

class CNN_BiLSTM_custom_pooling_dual_input_2(Model): # better than previous
    def __init__(self):
        super(CNN_BiLSTM_custom_pooling_dual_input_2, self).__init__()
        # DNA sequence stream
        self.conv1_seq = Conv1D(filters=100, kernel_size=10, padding='same')
        self.batchnorm1_seq = BatchNormalization()  # Batch normalization layer
        self.pool1_seq = CustomPooling(pool_size=2, strides=2)
        self.conv3_seq = Conv1D(filters=150, kernel_size=8, padding='same')
        self.batchnorm3_seq = BatchNormalization()  # Batch normalization layer
        self.drop2_seq = Dropout(0.2)

        # Gene/operon annotation stream
        self.conv1_anno = Conv1D(filters=50, kernel_size=4, padding='same')
        self.batchnorm1_anno = BatchNormalization()  # Batch normalization layer
        self.pool1_anno = CustomPooling(pool_size=2, strides=2)
        self.conv2_anno = Conv1D(filters=100, kernel_size=4, padding='same')
        self.batchnorm2_anno = BatchNormalization()  # Batch normalization layer
        self.drop2_anno = Dropout(0.2)

        # Shared layers
        self.bilstm = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))
        self.drop3 = Dropout(0.2)
        self.final_conv = Conv1D(filters=1, kernel_size=1, activation='relu', padding='same')

    def call(self, inputs):
        inputs_seq, inputs_anno = inputs
        # DNA sequence stream
        x_seq = self.conv1_seq(inputs_seq)
        x_seq = self.batchnorm1_seq(x_seq, training=False)
        x_seq = self.pool1_seq(x_seq)
        x_seq = self.conv3_seq(x_seq)
        x_seq = self.batchnorm3_seq(x_seq, training=False)
        x_seq = self.drop2_seq(x_seq)

        # Gene/operon annotation stream
        x_anno = self.conv1_anno(inputs_anno)
        x_anno = self.batchnorm1_anno(x_anno, training=False)
        x_anno = self.pool1_anno(x_anno)
        x_anno = self.conv2_anno(x_anno)
        x_anno = self.batchnorm2_anno(x_anno, training=False)
        x_anno = self.drop2_anno(x_anno)

        # Combine the outputs of both streams
        combined = concatenate([x_seq, x_anno])

        # Shared layers
        x = self.bilstm(combined)
        x = self.drop3(x)
        return self.final_conv(x)
    
class CNN_BiLSTM_custom_pooling_dual_input_4(Model): # better than previous
    def __init__(self):
        super(CNN_BiLSTM_custom_pooling_dual_input_4, self).__init__()
        # DNA sequence stream
        self.conv1_seq = Conv1D(filters=100, kernel_size=10, padding='same')
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
        self.conv2_anno = Conv1D(filters=100, kernel_size=4, padding='same')
        self.batchnorm2_anno = BatchNormalization()  # Batch normalization layer
        self.drop2_anno = Dropout(0.2)

        # Shared layers
        self.bilstm = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))
        self.drop3 = Dropout(0.2)
        self.final_conv = Conv1D(filters=1, kernel_size=1, activation='relu', padding='same')

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
    

class CNN_BiLSTM_custom_pooling_dual_input_4_2(Model): # better than previous
    def __init__(self):
        super(CNN_BiLSTM_custom_pooling_dual_input_4_2, self).__init__()
        # DNA sequence stream
        self.conv1_seq = Conv1D(filters=100, kernel_size=10, padding='same')
        self.batchnorm1_seq = BatchNormalization()  # Batch normalization layer
        self.pool1_seq = CustomPooling(pool_size=2, strides=2)
        self.pool2_seq = CustomPooling(pool_size=2, strides=2)
        self.conv3_seq = Conv1D(filters=150, kernel_size=20, padding='same')
        self.batchnorm3_seq = BatchNormalization()  # Batch normalization layer
        self.drop2_seq = Dropout(0.2)

        # Gene/operon annotation stream
        self.conv1_anno = Conv1D(filters=50, kernel_size=4, padding='same')
        self.batchnorm1_anno = BatchNormalization()  # Batch normalization layer
        self.pool1_anno = CustomPooling(pool_size=2, strides=2)
        self.pool2_anno = CustomPooling(pool_size=2, strides=2)
        self.conv2_anno = Conv1D(filters=100, kernel_size=16, padding='same')
        self.batchnorm2_anno = BatchNormalization()  # Batch normalization layer
        self.drop2_anno = Dropout(0.2)

        # Shared layers
        self.bilstm = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))
        self.drop3 = Dropout(0.2)
        self.final_conv = Conv1D(filters=1, kernel_size=1, activation='relu', padding='same')

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
    
class CNN_BiLSTM_custom_pooling_dual_input_4_3(Model): # better than previous
    def __init__(self):
        super(CNN_BiLSTM_custom_pooling_dual_input_4_3, self).__init__()
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
        self.final_conv = Conv1D(filters=1, kernel_size=1, activation='relu', padding='same')

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


class CNN_BiLSTM_avg_pooling_4_dual_input(Model):    #### avg pooling seems to fail
    def __init__(self):
        super(CNN_BiLSTM_avg_pooling_4_dual_input, self).__init__()
        # DNA sequence stream
        self.conv1_seq = Conv1D(filters=100, kernel_size=10, padding='same')
        self.batchnorm1_seq = BatchNormalization()  # Batch normalization layer
        self.pool1_seq = AveragePooling1D(pool_size=2, strides=2)
        self.pool2_seq = AveragePooling1D(pool_size=2, strides=2)
        self.conv3_seq = Conv1D(filters=150, kernel_size=8, padding='same')
        self.batchnorm3_seq = BatchNormalization()  # Batch normalization layer
        self.drop2_seq = Dropout(0.2)

        # Gene/operon annotation stream
        self.conv1_anno = Conv1D(filters=50, kernel_size=4, padding='same')
        self.batchnorm1_anno = BatchNormalization()  # Batch normalization layer
        self.pool1_anno = AveragePooling1D(pool_size=2, strides=2)
        self.pool2_anno = AveragePooling1D(pool_size=2, strides=2)
        self.conv2_anno = Conv1D(filters=100, kernel_size=4, padding='same')
        self.batchnorm2_anno = BatchNormalization()  # Batch normalization layer
        self.drop2_anno = Dropout(0.2)

        # Shared layers
        self.bilstm = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))
        self.batchnorm = BatchNormalization()
        self.drop3 = Dropout(0.2)
        self.final_conv = Conv1D(filters=1, kernel_size=1, activation='relu', padding='same')

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
        x = self.batchnorm(x)
        x = self.drop3(x)
        return self.final_conv(x)
    

class CNN_BiLSTM_avg_pooling_4_dual_input_2(Model):
    def __init__(self):
        super(CNN_BiLSTM_avg_pooling_4_dual_input_2, self).__init__()
        # DNA sequence stream
        self.conv1_seq = Conv1D(filters=100, kernel_size=10, padding='same')
        self.batchnorm1_seq = BatchNormalization()  # Batch normalization layer
        self.pool1_seq = AveragePooling1D(pool_size=2, strides=2)
        self.pool2_seq = AveragePooling1D(pool_size=2, strides=2)
        self.conv3_seq = Conv1D(filters=150, kernel_size=8, padding='same')
        self.batchnorm3_seq = BatchNormalization()  # Batch normalization layer
        self.drop2_seq = Dropout(0.2)

        # Gene/operon annotation stream
        self.conv1_anno = Conv1D(filters=50, kernel_size=4, padding='same')
        self.batchnorm1_anno = BatchNormalization()  # Batch normalization layer
        self.pool1_anno = AveragePooling1D(pool_size=2, strides=2)
        self.pool2_anno = AveragePooling1D(pool_size=2, strides=2)
        self.bilstm_anno = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))
        self.batchnorm2_anno = BatchNormalization()  # Batch normalization layer
        self.drop2_anno = Dropout(0.2)

        # Shared layers
        self.bilstm = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))
        self.batchnorm = BatchNormalization()
        self.drop3 = Dropout(0.2)
        self.final_conv = Conv1D(filters=1, kernel_size=1, activation='relu', padding='same')

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
        x_anno = self.bilstm_anno(x_anno)
        x_anno = self.batchnorm2_anno(x_anno, training=False)
        x_anno = self.drop2_anno(x_anno)

        # Combine the outputs of both streams
        combined = concatenate([x_seq, x_anno])

        # Shared layers
        x = self.bilstm(combined)
        x = self.batchnorm(x)
        x = self.drop3(x)
        return self.final_conv(x)
    
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