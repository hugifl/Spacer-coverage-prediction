import tensorflow as tf
from custom_elements import CustomPooling, AttentionMechanism
from keras.initializers import HeNormal
from tensorflow.keras.layers import Conv1D, ELU, BatchNormalization
from tensorflow.keras import backend as K

Model = tf.keras.models.Model
Conv1D = tf.keras.layers.Conv1D
MaxPooling1D = tf.keras.layers.MaxPooling1D
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense
Input = tf.keras.layers.Input
Dropout = tf.keras.layers.Dropout
Bidirectional = tf.keras.layers.Bidirectional
LSTM = tf.keras.layers.LSTM


class CNN_binary(Model):
    def __init__(self):
        super(CNN_binary, self).__init__()
        self.conv1 = Conv1D(filters=75, kernel_size=20, activation='relu', padding='same')
        self.pool1 = MaxPooling1D(pool_size=2, strides=2)
        self.pool2 = MaxPooling1D(pool_size=2, strides=2)
        self.pool3 = MaxPooling1D(pool_size=4, strides=4)
        self.final_conv = Conv1D(filters=1, kernel_size=1, activation='sigmoid')  

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.pool2(x)
        x = self.pool3(x)
        return self.final_conv(x)
    

class CNN_binary_dilated(Model):
    def __init__(self):
        super(CNN_binary_dilated, self).__init__()
        self.conv1 = Conv1D(filters=75, kernel_size=20, activation='relu', padding='same')
        self.conv2 = Conv1D(filters=64, kernel_size=3, dilation_rate=2, activation='relu', padding='same')
        self.conv3 = Conv1D(filters=64, kernel_size=3, dilation_rate=4, activation='relu', padding='same')
        self.conv4 = Conv1D(filters=64, kernel_size=3, dilation_rate=8, activation='relu', padding='same')
        self.conv5 = Conv1D(filters=64, kernel_size=3, dilation_rate=16, activation='relu', padding='same')
        self.pool1 = MaxPooling1D(pool_size=2, strides=2)
        self.pool2 = MaxPooling1D(pool_size=2, strides=2)
        self.pool3 = MaxPooling1D(pool_size=4, strides=4)
        self.final_conv = Conv1D(filters=1, kernel_size=1, activation='sigmoid', padding='same')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool1(x)
        x = self.pool2(x)
        x = self.pool3(x)
        return self.final_conv(x)
    
class CNN_binary_BiLSTM_custom_pooling(Model):
    def __init__(self):
        super(CNN_binary_BiLSTM_custom_pooling, self).__init__()
        self.conv1 = Conv1D(filters=150, kernel_size=12, activation='relu', padding='same') #  125 10
        self.pool1 = CustomPooling(pool_size=2, strides=2)
        self.pool2 = CustomPooling(pool_size=2, strides=2)
        self.pool3 = CustomPooling(pool_size=4, strides=4)
        self.bilstm = Bidirectional(LSTM(32, return_sequences=True))
        self.conv2 = Conv1D(filters=100, kernel_size=5, activation='relu', padding='same') #  125 10
        self.final_conv = Conv1D(filters=1, kernel_size=1, activation='sigmoid', padding='same')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.pool2(x)
        x = self.pool3(x)
        x = self.bilstm(x)
        #x = self.conv2(x)
        return self.final_conv(x)

class CNN_binary_BiLSTM_custom_pooling_2(Model):
    def __init__(self):
        super(CNN_binary_BiLSTM_custom_pooling_2, self).__init__()
        self.conv1 = Conv1D(filters=75, kernel_size=12, activation='relu', padding='same') #  125 10
        self.conv2 = Conv1D(filters=150, kernel_size=6, activation='relu', padding='same') #  125 10
        self.pool1 = CustomPooling(pool_size=2, strides=2)
        self.pool2 = CustomPooling(pool_size=2, strides=2)
        self.pool3 = CustomPooling(pool_size=4, strides=4)
        self.bilstm = Bidirectional(LSTM(32, return_sequences=True))
        self.final_conv = Conv1D(filters=1, kernel_size=1, activation='sigmoid', padding='same')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.pool2(x)
        x = self.pool3(x)
        x = self.bilstm(x)
        #x = self.conv2(x)
        return self.final_conv(x)
    
class CNN_binary_BiLSTM_custom_pooling_3(Model):
    def __init__(self):
        super(CNN_binary_BiLSTM_custom_pooling_3, self).__init__()
        self.conv1 = Conv1D(filters=75, kernel_size=12, activation='relu', padding='same') #  125 10
        self.conv2 = Conv1D(filters=150, kernel_size=6, activation='relu', padding='same') #  125 10
        self.pool1 = CustomPooling(pool_size=2, strides=2)
        self.bilstm = Bidirectional(LSTM(32, return_sequences=True))
        self.final_conv = Conv1D(filters=1, kernel_size=1, activation='sigmoid', padding='same')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.bilstm(x)
        #x = self.conv2(x)
        return self.final_conv(x)

class CNN_binary_BiLSTM_attention_custom_pooling(Model):
    def __init__(self):
        super(CNN_binary_BiLSTM_attention_custom_pooling, self).__init__()
        self.conv1 = Conv1D(filters=125, kernel_size=10, activation='relu', padding='same')
        self.pool1 = CustomPooling(pool_size=2, strides=2)
        self.pool2 = CustomPooling(pool_size=2, strides=2)
        self.pool3 = CustomPooling(pool_size=4, strides=4)
        self.attention = AttentionMechanism()
        self.dropout1 = Dropout(0.2)
        self.bilstm = Bidirectional(LSTM(32, return_sequences=True))
        self.conv2 = Conv1D(filters=125, kernel_size=10, activation='relu', padding='same')
        self.final_conv = Conv1D(filters=1, kernel_size=1, activation='sigmoid', padding='same')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.pool2(x)
        x = self.pool3(x)
        x = self.attention(x)
        x = self.dropout1(x)
        x = self.bilstm(x)
        x = self.conv2(x)
        return self.final_conv(x)
    

    ################################################# COVERAGE PREDICTION #######################################################################

class CNN_BiLSTM_custom_pooling_coverage(Model):
    def __init__(self):
        super(CNN_BiLSTM_custom_pooling_coverage, self).__init__()
        self.conv1 = Conv1D(filters=75, kernel_size=12, padding='same', kernel_initializer=HeNormal())
        self.batch_norm1 = BatchNormalization()
        self.elu1 = ELU(alpha=1.0)
        self.conv2 = Conv1D(filters=150, kernel_size=6, padding='same', kernel_initializer=HeNormal())
        self.batch_norm2 = BatchNormalization()
        self.elu2 = ELU(alpha=1.0)
        self.pool1 = CustomPooling(pool_size=2, strides=2)
        self.pool2 = CustomPooling(pool_size=2, strides=2)
        self.pool3 = CustomPooling(pool_size=4, strides=4)
        self.bilstm = Bidirectional(LSTM(32, return_sequences=True))
        self.final_conv = Conv1D(filters=1, kernel_size=1, activation='linear', padding='same')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.batch_norm1(x)
        x = self.elu1(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.elu2(x)
        x = self.pool1(x)
        x = self.pool2(x)
        x = self.pool3(x)
        x = self.bilstm(x)
        return self.final_conv(x)
    
class Simple_CNN_custom_pooling_coverage(Model):
    def __init__(self):
        super(Simple_CNN_custom_pooling_coverage, self).__init__()
        self.conv1 = Conv1D(filters=75, kernel_size=12, padding='same', kernel_initializer=HeNormal())
        self.batch_norm1 = BatchNormalization()
        self.elu1 = ELU(alpha=1.0)
        self.pool1 = CustomPooling(pool_size=2, strides=2)
        self.pool2 = CustomPooling(pool_size=2, strides=2)
        self.pool3 = CustomPooling(pool_size=4, strides=4)
        self.final_conv = Conv1D(filters=1, kernel_size=1, activation='linear', padding='same')

    def call(self, inputs):
        x = self.conv1(inputs)
        print("After conv1: Min:", K.min(x), "Max:", K.max(x), "NaNs:", K.any(tf.math.is_nan(x)))
        x = self.batch_norm1(x)
        x = self.elu1(x)
        x = self.pool1(x)
        x = self.pool2(x)
        x = self.pool3(x)
        return self.final_conv(x)
    

class CNN_BiLSTM_custom_pooling_2(Model):
    def __init__(self):
        super(CNN_BiLSTM_custom_pooling_2, self).__init__()
        self.conv1 = Conv1D(filters=75, kernel_size=12, activation='relu', padding='same') #  125 10
        self.conv2 = Conv1D(filters=150, kernel_size=6, activation='relu', padding='same') #  125 10
        self.pool1 = CustomPooling(pool_size=2, strides=2)
        self.pool2 = CustomPooling(pool_size=2, strides=2)
        self.pool3 = CustomPooling(pool_size=4, strides=4)
        self.bilstm = Bidirectional(LSTM(32, return_sequences=True))
        self.final_conv = Conv1D(filters=1, kernel_size=1, activation='linear', padding='same')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.pool2(x)
        x = self.pool3(x)
        x = self.bilstm(x)
        return self.final_conv(x)
    

class CNN_custom_pooling(Model):
    def __init__(self):
        super(CNN_custom_pooling, self).__init__()
        self.conv1 = Conv1D(filters=75, kernel_size=12, activation='relu', padding='same') #  125 10
        self.conv2 = Conv1D(filters=150, kernel_size=6, activation='relu', padding='same') #  125 10
        self.pool1 = CustomPooling(pool_size=2, strides=2)
        self.pool2 = CustomPooling(pool_size=2, strides=2)
        self.pool3 = CustomPooling(pool_size=4, strides=4)
        self.final_conv = Conv1D(filters=1, kernel_size=1, activation='linear', padding='same')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.pool2(x)
        x = self.pool3(x)
        return self.final_conv(x)
    

class CNN_BiLSTM_custom_pooling_2_poisson(Model):
    def __init__(self):
        super(CNN_BiLSTM_custom_pooling_2_poisson, self).__init__()
        self.conv1 = Conv1D(filters=100, kernel_size=4, activation='relu', padding='same') #  75 12
        #self.drop1 = Dropout(0.2)
        self.pool1 = CustomPooling(pool_size=2, strides=2)
        self.conv2 = Conv1D(filters=200, kernel_size=4, activation='relu', padding='same') #  150 6
        self.drop2 = Dropout(0.2)
        self.pool2 = CustomPooling(pool_size=2, strides=2)
        self.pool3 = CustomPooling(pool_size=4, strides=4)
        self.bilstm = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))
        self.drop3 = Dropout(0.2)
        self.final_conv = Conv1D(filters=1, kernel_size=1, activation='relu', padding='same')

    def call(self, inputs):
        x = self.conv1(inputs)
        #x = self.drop1(x)   
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.drop2(x)
        x = self.pool2(x)
        x = self.pool3(x)
        x = self.bilstm(x)
        x = self.drop3(x)
        return self.final_conv(x)
    
class CNN_BiLSTM_custom_pooling_attention_poisson(Model):
    def __init__(self):
        super(CNN_BiLSTM_custom_pooling_attention_poisson, self).__init__()
        self.conv1 = Conv1D(filters=75, kernel_size=12, activation='relu', padding='same') #  125 10
        self.drop1 = Dropout(0.2)
        #self.conv2 = Conv1D(filters=150, kernel_size=6, activation='relu', padding='same') #  125 10
        #self.drop2 = Dropout(0.2)
        self.pool1 = CustomPooling(pool_size=2, strides=2)
        self.pool2 = CustomPooling(pool_size=2, strides=2)
        self.pool3 = CustomPooling(pool_size=4, strides=4)
        self.attention = AttentionMechanism()
        self.bilstm = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))
        self.final_conv = Conv1D(filters=1, kernel_size=1, activation='linear', padding='same')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.drop1(x)   
        #x = self.conv2(x)
        #x = self.drop2(x)
        x = self.pool1(x)
        x = self.pool2(x)
        x = self.pool3(x)
        x = self.attention(x)
        x = self.bilstm(x)
        return self.final_conv(x)
    

class CNN_BiLSTM_custom_pooling_poisson_bin32(Model):
    def __init__(self):
        super(CNN_BiLSTM_custom_pooling_poisson_bin32, self).__init__()
        self.conv1 = Conv1D(filters=100, kernel_size=4, activation='relu', padding='same') #  75 12
        #self.drop1 = Dropout(0.2)
        self.pool1 = CustomPooling(pool_size=2, strides=2)
        self.conv2 = Conv1D(filters=200, kernel_size=4, activation='relu', padding='same') #  150 6
        self.drop2 = Dropout(0.2)
        self.pool2 = CustomPooling(pool_size=2, strides=2)
        self.pool3 = CustomPooling(pool_size=4, strides=2)
        self.pool4 = CustomPooling(pool_size=4, strides=4)
        self.bilstm = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))
        self.drop3 = Dropout(0.2)
        self.final_conv = Conv1D(filters=1, kernel_size=1, activation='relu', padding='same')

    def call(self, inputs):
        x = self.conv1(inputs)
        #x = self.drop1(x)   
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.drop2(x)
        x = self.pool2(x)
        x = self.pool3(x)
        x = self.pool4(x)   
        x = self.bilstm(x)
        x = self.drop3(x)
        return self.final_conv(x)
    
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