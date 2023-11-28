import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

# Pooling layer that applies max pooling on all channels and max pooling of the absolute values on the last channel (contains -1/1 values)
class CustomPooling(tf.keras.layers.Layer):
    def __init__(self, pool_size, strides, padding='SAME', **kwargs):
        super(CustomPooling, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.large_positive_value = tf.constant(2.0, dtype=tf.float32)  # Ensure it's a float

    def call(self, inputs):
        # Separate the 7th channel
        channel_7 = inputs[..., -1:]

        # Replace -1 with a large positive value
        replaced_channel_7 = tf.where(channel_7 == -1, self.large_positive_value, channel_7)

        # Apply max pooling on all channels except the last
        pooled = tf.nn.max_pool(inputs[..., :-1], ksize=[1, self.pool_size, 1], strides=[1, self.strides, 1], padding=self.padding)

        # Apply max pooling on the 7th channel with replaced values
        pooled_channel_7 = tf.nn.max_pool(replaced_channel_7, ksize=[1, self.pool_size, 1], strides=[1, self.strides, 1], padding=self.padding)

        # Restore the original sign in the 7th channel
        restored_channel_7 = tf.where(pooled_channel_7 == self.large_positive_value, tf.constant(-1.0, dtype=tf.float32), pooled_channel_7)

        # Concatenate pooled channels and the 7th channel
        output = tf.concat([pooled, restored_channel_7], axis=-1)
        return output
    
# Custom binary cross entropy loss function for binary peak prediction that punishes false positives more than false negatives and does l1 regularization.
def custom_loss_with_l1(y_true, y_pred):
    y_true = tf.cast(y_true, y_pred.dtype)

    # Manually calculate BCE for each position
    bce = -(y_true * tf.math.log(y_pred + 1e-15) + (1 - y_true) * tf.math.log(1 - y_pred + 1e-15))

    # Define weight for false negatives
    false_negative_weight = 5.0

    # Apply weights
    weight = tf.where(tf.less(y_pred, y_true), false_negative_weight, 1.0)
    weighted_bce = bce * weight

    # Sum up the weighted BCEs for each sample in the batch
    weighted_bce_sum = tf.reduce_sum(weighted_bce, axis=-1)

    # L1 Regularization
    l1_lambda = 0.01
    l1_reg = l1_lambda * tf.reduce_sum(tf.abs(y_pred))

    # Combine BCE and L1 regularization
    combined_loss = tf.reduce_mean(weighted_bce_sum) + l1_reg

    return combined_loss

# Attention mechanism that applies softmax on the dot product of the activation map and a learnable key vector  
class AttentionMechanism(Layer):
    def __init__(self, **kwargs):
        super(AttentionMechanism, self).__init__(**kwargs)

    def build(self, input_shape):
        self.key_vector = self.add_weight(name='key_vector',
                                          shape=(input_shape[2], 1),
                                          initializer='uniform',
                                          trainable=True)
        super(AttentionMechanism, self).build(input_shape)

    def call(self, activation_map):
        # Dot product with key vector and apply softmax
        attention_weights = tf.nn.softmax(tf.matmul(activation_map, self.key_vector), axis=1)
        # Scale activation map
        scaled_activation_map = activation_map * attention_weights
        return scaled_activation_map

    def compute_output_shape(self, input_shape):
        return input_shape


# Custom poisson loss function that avoids NaNs and Infs
def poisson_loss(y_true, y_pred):
    return K.mean(K.maximum(.0, y_pred) - y_true * K.log(K.maximum(.0, y_pred) + K.epsilon()), axis=-1)

    