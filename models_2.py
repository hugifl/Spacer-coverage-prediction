import tensorflow as tf
from custom_elements import CustomPooling, Attention
from keras.initializers import HeNormal
from tensorflow.keras.layers import Conv1D, ELU, BatchNormalization, AveragePooling1D, AdditiveAttention,  concatenate, MultiHeadAttention, LayerNormalization,  Input, Conv1D, Dropout, Bidirectional, LSTM, concatenate, Dense, Flatten, Activation, RepeatVector, Permute, Multiply, Lambda, Layer
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



class CNN_BiLSTM_custom_pooling_dual_input_old(Model): # better than previous
    def __init__(self):
        super(CNN_BiLSTM_custom_pooling_dual_input_old, self).__init__()
        # DNA sequence stream
        self.conv1_seq = Conv1D(filters=100, kernel_size=10, padding='same')
        self.batchnorm1_seq = BatchNormalization()  # Batch normalization layer
        self.conv3_seq = Conv1D(filters=150, kernel_size=20, padding='same')
        self.batchnorm3_seq = BatchNormalization()  # Batch normalization layer
        self.drop2_seq = Dropout(0.2)

        # Gene/operon annotation stream
        self.conv1_anno = Conv1D(filters=50, kernel_size=4, padding='same')
        self.batchnorm1_anno = BatchNormalization()  # Batch normalization layer
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
        x_seq = self.conv3_seq(x_seq)
        x_seq = self.batchnorm3_seq(x_seq, training=False)
        x_seq = self.drop2_seq(x_seq)

        # Gene/operon annotation stream
        x_anno = self.conv1_anno(inputs_anno)
        x_anno = self.batchnorm1_anno(x_anno, training=False)
        x_anno = self.conv2_anno(x_anno)
        x_anno = self.batchnorm2_anno(x_anno, training=False)
        x_anno = self.drop2_anno(x_anno)

        # Combine the outputs of both streams
        combined = concatenate([x_seq, x_anno])

        # Shared layers
        x = self.bilstm(combined)
        x = self.drop3(x)
        return self.final_conv(x)


class CNN_binned(Model):
    def __init__(self, num_layers_seq, num_layers_anno, filter_number_seq, filter_number_anno, kernel_size_seq, kernel_size_anno, only_seq, input_dim, output_dim):
        super(CNN_binned, self).__init__()

        # Store num_layers_seq as an instance attribute
        self.num_layers_seq = num_layers_seq
        self.num_layers_anno = num_layers_anno
        self.only_seq = only_seq

        # Assertions for hyperparameter list lengths
        assert len(filter_number_seq) >= num_layers_seq and len(kernel_size_seq) >= num_layers_seq, "Sequence stream lists must match the number of layers"
        assert len(filter_number_anno) >= num_layers_anno and len(kernel_size_anno) >= num_layers_anno, "Annotation stream lists must match the number of layers"

        # Calculate pool size for binning
        self.pool_size = input_dim // output_dim
        self.stride = self.pool_size  # Assuming the stride equals the pool size for non-overlapping pooling

        # Initial pooling layer for single-layer CNN scenario
        self.initial_pooling = AveragePooling1D(pool_size=self.pool_size, strides=self.stride, padding='same')

        # DNA sequence stream layers
        self.seq_layers = []
        if num_layers_seq == 1:
            # For a single-layer CNN, the pooling happens before the CNN layer
            self.seq_layers.append(self.initial_pooling)
        self.seq_layers.extend([Conv1D(filters=filter_number_seq[i], kernel_size=kernel_size_seq[i], padding='same', activation='relu') for i in range(num_layers_seq)])

        self.drop_seq = Dropout(0.2)

        # Annotation stream layers, if applicable
        self.anno_layers = []
        if not only_seq:
            if num_layers_anno == 1:
                # For a single-layer CNN, the pooling happens before the CNN layer
                self.anno_layers.append(self.initial_pooling)
            self.anno_layers.extend([Conv1D(filters=filter_number_anno[i], kernel_size=kernel_size_anno[i], padding='same', activation='relu') for i in range(num_layers_anno)])
            self.drop_anno = Dropout(0.2)

        # Shared layers
        self.final_conv = Conv1D(filters=1, kernel_size=1, activation='relu', padding='same')

    def call(self, inputs):
        inputs_seq, inputs_anno = inputs

        # Apply binning directly to sequence input if num_layers_seq == 1 or after first conv layer if num_layers_seq > 1
        if self.num_layers_seq == 1:
            x_seq = self.seq_layers[0](inputs_seq)
            x_seq = self.seq_layers[1](x_seq)
        else:
            x_seq = self.seq_layers[0](inputs_seq)
            x_seq = self.initial_pooling(x_seq)
            for layer in self.seq_layers[1:]:
                x_seq = layer(x_seq)
        x_seq = self.drop_seq(x_seq)

        if not self.only_seq:
            if self.num_layers_anno == 1:
                x_anno = self.anno_layers[0](inputs_anno)
                x_anno = self.anno_layers[1](x_anno)
            else:
                x_anno = self.anno_layers[0](inputs_anno)
                x_anno = self.initial_pooling(x_anno)
                for layer in self.anno_layers[1:]:
                    x_anno = layer(x_anno)
            x_anno = self.drop_anno(x_anno)
            # Ensure shapes align for concatenation
            combined = concatenate([x_seq, x_anno], axis=-1)
        else:
            combined = x_seq

        return self.final_conv(combined)

class biLSTM(Model):
    def __init__(self, num_layers_seq, num_layers_anno, unit_numbers_seq, unit_numbers_anno, only_seq):
        super(biLSTM, self).__init__()

        # Assertions for hyperparameter list lengths
        assert len(unit_numbers_seq) >= num_layers_seq, "Sequence stream lists must match the number of layers"
        if not only_seq:
            assert len(unit_numbers_anno) >= num_layers_anno, "Annotation stream lists must match the number of layers"

        self.only_seq = only_seq  

        # DNA sequence stream layers
        self.seq_layers = [Bidirectional(LSTM(unit_numbers_seq[i], return_sequences=True)) for i in range(num_layers_seq)]
        self.drop_seq = Dropout(0.2)

        # Annotation stream layers, if applicable
        self.anno_layers = []
        if not only_seq:
            self.anno_layers = [Bidirectional(LSTM(unit_numbers_anno[i], return_sequences=True)) for i in range(num_layers_anno)]
            self.drop_anno = Dropout(0.2)

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

            # Combine sequence and annotation streams
            combined = concatenate([x_seq, x_anno], axis=-1)
        else:
            combined = x_seq

        # Shared layers
        return self.final_conv(combined)

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


class CNN_biLSTM_1(Model):
    def __init__(self, CNN_num_layers_seq, CNN_num_layers_anno, filter_number_seq, filter_number_anno, kernel_size_seq, kernel_size_anno, biLSTM_num_layers_seq, biLSTM_num_layers_anno, unit_numbers_seq, unit_numbers_anno, unit_numbers_combined, only_seq):
        super(CNN_biLSTM_1, self).__init__()

        assert len(filter_number_seq) >= CNN_num_layers_seq and len(kernel_size_seq) >= CNN_num_layers_seq, "Sequence stream lists must match the number of layers"
        assert len(filter_number_anno) >= CNN_num_layers_anno and len(kernel_size_anno) >= CNN_num_layers_anno, "Annotation stream lists must match the number of layers"

        assert len(unit_numbers_seq) >= biLSTM_num_layers_seq, "Sequence stream lists must match the number of layers"
        assert len(unit_numbers_anno) >= biLSTM_num_layers_anno, "Annotation stream lists must match the number of layers"

        # DNA sequence stream CNN layers
        self.seq_layers_CNN = [Conv1D(filters=filter_number_seq[i], kernel_size=kernel_size_seq[i], padding='same', activation='relu') for i in range(CNN_num_layers_seq)]
        self.drop_seq = Dropout(0.2)

        self.only_seq = only_seq  
        self.unit_numbers_combined = unit_numbers_combined

        if not only_seq:
            # Gene/operon annotation stream CNN layers
            self.anno_layers_CNN = [Conv1D(filters=filter_number_anno[i], kernel_size=kernel_size_anno[i], padding='same', activation='relu') for i in range(CNN_num_layers_anno)]
            self.drop_anno = Dropout(0.2)

        # DNA sequence stream biLSTM layers
        self.seq_layers_biLSTM = [Bidirectional(LSTM(unit_numbers_seq[i], return_sequences=True)) for i in range(biLSTM_num_layers_seq)]

        if not only_seq:
            self.anno_layers_biLSTM = [Bidirectional(LSTM(unit_numbers_anno[i], return_sequences=True)) for i in range(biLSTM_num_layers_anno)]

        # Shared layers
        if unit_numbers_combined != 0:
            self.final_biLSTM = Bidirectional(LSTM(unit_numbers_combined, return_sequences=True))
        self.final_conv = Conv1D(filters=1, kernel_size=1, activation='relu', padding='same')

    def call(self, inputs):
        inputs_seq, inputs_anno = inputs

        # DNA sequence stream
        x_seq = inputs_seq
        for layer in self.seq_layers_CNN:
            x_seq = layer(x_seq)
        
        for layer in self.seq_layers_biLSTM:
            x_seq = layer(x_seq)  
        x_seq = self.drop_seq(x_seq)

        if not self.only_seq:
            # Gene/operon annotation stream
            x_anno = inputs_anno
            for layer in self.anno_layers_CNN:
                x_anno = layer(x_anno)
            for layer in self.anno_layers_biLSTM:
                x_anno = layer(x_anno)  
            x_anno = self.drop_anno(x_anno)
            combined = concatenate([x_seq, x_anno])
        else:
            combined = x_seq

        # Shared layers
        if self.unit_numbers_combined != 0:
            combined = self.final_biLSTM(combined)
        return self.final_conv(combined)
    
class CNN_biLSTM_AddAttention(Model):
    def __init__(self, CNN_num_layers_seq, CNN_num_layers_anno, filter_number_seq, filter_number_anno, kernel_size_seq, kernel_size_anno, biLSTM_num_layers_seq, biLSTM_num_layers_anno, unit_numbers_seq, unit_numbers_anno, unit_numbers_combined, only_seq):
        super(CNN_biLSTM_AddAttention, self).__init__()

        assert len(filter_number_seq) >= CNN_num_layers_seq and len(kernel_size_seq) >= CNN_num_layers_seq, "Sequence stream lists must match the number of layers"
        assert len(filter_number_anno) >= CNN_num_layers_anno and len(kernel_size_anno) >= CNN_num_layers_anno, "Annotation stream lists must match the number of layers"

        assert len(unit_numbers_seq) >= biLSTM_num_layers_seq, "Sequence stream lists must match the number of layers"
        assert len(unit_numbers_anno) >= biLSTM_num_layers_anno, "Annotation stream lists must match the number of layers"

        # DNA sequence stream CNN layers
        self.seq_layers_CNN = [Conv1D(filters=filter_number_seq[i], kernel_size=kernel_size_seq[i], padding='same', activation='relu') for i in range(CNN_num_layers_seq)]
        self.drop_seq = Dropout(0.2)

        self.only_seq = only_seq  
        self.unit_numbers_combined = unit_numbers_combined

        if not only_seq:
            # Gene/operon annotation stream CNN layers
            self.anno_layers_CNN = [Conv1D(filters=filter_number_anno[i], kernel_size=kernel_size_anno[i], padding='same', activation='relu') for i in range(CNN_num_layers_anno)]
            self.drop_anno = Dropout(0.2)

        # DNA sequence stream biLSTM layers
        self.seq_layers_biLSTM = [Bidirectional(LSTM(unit_numbers_seq[i], return_sequences=True)) for i in range(biLSTM_num_layers_seq)]

        if not only_seq:
            self.anno_layers_biLSTM = [Bidirectional(LSTM(unit_numbers_anno[i], return_sequences=True)) for i in range(biLSTM_num_layers_anno)]

        # Shared layers
        if unit_numbers_combined != 0:
            self.final_biLSTM = Bidirectional(LSTM(unit_numbers_combined, return_sequences=True))
        self.attention = AdditiveAttention(use_scale=True)
        self.final_conv = Conv1D(filters=1, kernel_size=1, activation='relu', padding='same')

    def call(self, inputs):
        inputs_seq, inputs_anno = inputs

        # DNA sequence stream
        x_seq = inputs_seq
        for layer in self.seq_layers_CNN:
            x_seq = layer(x_seq)
        
        for layer in self.seq_layers_biLSTM:
            x_seq = layer(x_seq)  
        x_seq = self.drop_seq(x_seq)

        if not self.only_seq:
            # Gene/operon annotation stream
            x_anno = inputs_anno
            for layer in self.anno_layers_CNN:
                x_anno = layer(x_anno)
            for layer in self.anno_layers_biLSTM:
                x_anno = layer(x_anno)  
            x_anno = self.drop_anno(x_anno)
            combined = concatenate([x_seq, x_anno])
        else:
            combined = x_seq

        # Shared layers
        if self.unit_numbers_combined != 0:
            combined = self.final_biLSTM(combined)
        combined = self.attention([combined, combined])
        return self.final_conv(combined)
    
class CNN_biLSTM_CustomAttention(Model):
    def __init__(self, CNN_num_layers_seq, CNN_num_layers_anno, filter_number_seq, filter_number_anno, kernel_size_seq, kernel_size_anno, biLSTM_num_layers_seq, biLSTM_num_layers_anno, unit_numbers_seq, unit_numbers_anno, unit_numbers_combined, only_seq):
        super(CNN_biLSTM_CustomAttention, self).__init__()

        assert len(filter_number_seq) >= CNN_num_layers_seq and len(kernel_size_seq) >= CNN_num_layers_seq, "Sequence stream lists must match the number of layers"
        assert len(filter_number_anno) >= CNN_num_layers_anno and len(kernel_size_anno) >= CNN_num_layers_anno, "Annotation stream lists must match the number of layers"

        assert len(unit_numbers_seq) >= biLSTM_num_layers_seq, "Sequence stream lists must match the number of layers"
        assert len(unit_numbers_anno) >= biLSTM_num_layers_anno, "Annotation stream lists must match the number of layers"

        # DNA sequence stream CNN layers
        self.seq_layers_CNN = [Conv1D(filters=filter_number_seq[i], kernel_size=kernel_size_seq[i], padding='same', activation='relu') for i in range(CNN_num_layers_seq)]
        self.drop_seq = Dropout(0.2)

        self.only_seq = only_seq  
        self.unit_numbers_combined = unit_numbers_combined

        if not only_seq:
            # Gene/operon annotation stream CNN layers
            self.anno_layers_CNN = [Conv1D(filters=filter_number_anno[i], kernel_size=kernel_size_anno[i], padding='same', activation='relu') for i in range(CNN_num_layers_anno)]
            self.drop_anno = Dropout(0.2)

        # DNA sequence stream biLSTM layers
        self.seq_layers_biLSTM = [Bidirectional(LSTM(unit_numbers_seq[i], return_sequences=True)) for i in range(biLSTM_num_layers_seq)]

        if not only_seq:
            self.anno_layers_biLSTM = [Bidirectional(LSTM(unit_numbers_anno[i], return_sequences=True)) for i in range(biLSTM_num_layers_anno)]

        # Custom attention mechanism
        self.attention_dense = Dense(1, activation='tanh')
        self.attention_flatten = Flatten()
        self.attention_activation = Activation('softmax')
        self.attention_repeat = RepeatVector(unit_numbers_combined * 2)  # Adjust based on your model's specifics
        self.attention_permute = Permute([2, 1])
        self.attention_multiply = Multiply()
        self.attention_lambda = Lambda(lambda values: K.sum(values, axis=1), output_shape=(unit_numbers_combined * 2,))

        # Shared layers
        if unit_numbers_combined != 0:
            self.final_biLSTM = Bidirectional(LSTM(unit_numbers_combined, return_sequences=True))
        self.final_conv = Conv1D(filters=1, kernel_size=1, activation='relu', padding='same')

    def call(self, inputs):
        inputs_seq, inputs_anno = inputs

        # DNA sequence stream
        x_seq = inputs_seq
        for layer in self.seq_layers_CNN:
            x_seq = layer(x_seq)
        
        for layer in self.seq_layers_biLSTM:
            x_seq = layer(x_seq)  
        x_seq = self.drop_seq(x_seq)

        if not self.only_seq:
            # Gene/operon annotation stream
            x_anno = inputs_anno
            for layer in self.anno_layers_CNN:
                x_anno = layer(x_anno)
            for layer in self.anno_layers_biLSTM:
                x_anno = layer(x_anno)  
            x_anno = self.drop_anno(x_anno)
            combined = concatenate([x_seq, x_anno])
        else:
            combined = x_seq
        # Shared layers
        if self.unit_numbers_combined != 0:
            combined = self.final_biLSTM(combined)
        attention_weighted = self.apply_attention(combined)

        return self.final_conv(attention_weighted)
    
    def apply_attention(self, lstm_out):
        e = self.attention_dense(lstm_out)
        a = self.attention_activation(e)
        output = lstm_out * a  # Element-wise multiplication to apply weights
        return output
    
class CNN_biLSTM_CustomAttention_2(Model):
    def __init__(self, CNN_num_layers_seq, CNN_num_layers_anno, filter_number_seq, filter_number_anno, kernel_size_seq, kernel_size_anno, biLSTM_num_layers_seq, biLSTM_num_layers_anno, unit_numbers_seq, unit_numbers_anno, unit_numbers_combined, only_seq):
        super(CNN_biLSTM_CustomAttention_2, self).__init__()

        assert len(filter_number_seq) >= CNN_num_layers_seq and len(kernel_size_seq) >= CNN_num_layers_seq, "Sequence stream lists must match the number of layers"
        assert len(filter_number_anno) >= CNN_num_layers_anno and len(kernel_size_anno) >= CNN_num_layers_anno, "Annotation stream lists must match the number of layers"

        assert len(unit_numbers_seq) >= biLSTM_num_layers_seq, "Sequence stream lists must match the number of layers"
        assert len(unit_numbers_anno) >= biLSTM_num_layers_anno, "Annotation stream lists must match the number of layers"

        # DNA sequence stream CNN layers
        self.seq_layers_CNN = [Conv1D(filters=filter_number_seq[i], kernel_size=kernel_size_seq[i], padding='same', activation='relu') for i in range(CNN_num_layers_seq)]
        self.drop_seq = Dropout(0.2)

        self.only_seq = only_seq  
        self.unit_numbers_combined = unit_numbers_combined

        if not only_seq:
            # Gene/operon annotation stream CNN layers
            self.anno_layers_CNN = [Conv1D(filters=filter_number_anno[i], kernel_size=kernel_size_anno[i], padding='same', activation='relu') for i in range(CNN_num_layers_anno)]
            self.drop_anno = Dropout(0.2)

        # DNA sequence stream biLSTM layers
        self.seq_layers_biLSTM = [Bidirectional(LSTM(unit_numbers_seq[i], return_sequences=True)) for i in range(biLSTM_num_layers_seq)]

        if not only_seq:
            self.anno_layers_biLSTM = [Bidirectional(LSTM(unit_numbers_anno[i], return_sequences=True)) for i in range(biLSTM_num_layers_anno)]

        # Integrating the custom Attention layer
        self.attention_layer = Attention(return_sequences=True)

        # Shared layers
        if unit_numbers_combined != 0:
            self.final_biLSTM = Bidirectional(LSTM(unit_numbers_combined, return_sequences=True))

        self.final_conv = Conv1D(filters=1, kernel_size=1, activation='relu', padding='same')

    def call(self, inputs):
        inputs_seq, inputs_anno = inputs

        # DNA sequence stream
        x_seq = inputs_seq
        for layer in self.seq_layers_CNN:
            x_seq = layer(x_seq)
        
        for layer in self.seq_layers_biLSTM:
            x_seq = layer(x_seq)  
        x_seq = self.drop_seq(x_seq)

        if not self.only_seq:
            # Gene/operon annotation stream
            x_anno = inputs_anno
            for layer in self.anno_layers_CNN:
                x_anno = layer(x_anno)
            for layer in self.anno_layers_biLSTM:
                x_anno = layer(x_anno)  
            x_anno = self.drop_anno(x_anno)
            combined = concatenate([x_seq, x_anno])
        else:
            combined = x_seq

        # Shared layers
        if self.unit_numbers_combined != 0:
            combined = self.final_biLSTM(combined)

        attention_output = self.attention_layer(combined)

        return self.final_conv(attention_output)
    

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