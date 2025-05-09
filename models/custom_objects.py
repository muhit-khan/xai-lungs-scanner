import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Lambda, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, Reshape, Add, Activation, Multiply, Concatenate
from tensorflow.keras import backend as K

class ChannelAttention(Layer):
    def __init__(self, filters, ratio=8, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.filters = filters
        self.ratio = ratio
        # Use He initializer for ReLU
        self.shared_dense_one = Dense(filters // ratio,
                                     activation='relu',
                                     kernel_initializer='he_normal',
                                     use_bias=True,
                                     bias_initializer='zeros',
                                     name='ca_dense_1')
        self.shared_dense_two = Dense(filters,
                                     kernel_initializer='he_normal', # Or Glorot for linear/sigmoid
                                     use_bias=True,
                                     bias_initializer='zeros',
                                     name='ca_dense_2')

    def call(self, inputs):
        avg_pool = GlobalAveragePooling2D(name='ca_avg_pool')(inputs)
        avg_pool = Reshape((1, 1, self.filters), name='ca_reshape_avg')(avg_pool)
        assert avg_pool.shape[1:] == (1, 1, self.filters)
        avg_pool = self.shared_dense_one(avg_pool)
        avg_pool = self.shared_dense_two(avg_pool)

        # Use K.max which is compatible with TF graph execution
        max_pool = Lambda(lambda x: K.max(x, axis=[1, 2], keepdims=True), name='ca_max_pool')(inputs)
        assert max_pool.shape[1:] == (1, 1, self.filters)
        max_pool = self.shared_dense_one(max_pool)
        max_pool = self.shared_dense_two(max_pool)

        attention = Add(name='ca_add')([avg_pool, max_pool])
        attention = Activation('sigmoid', name='ca_sigmoid')(attention)

        # Multiply attention scores with input feature map
        return Multiply(name='ca_multiply')([inputs, attention])

    def get_config(self):
        config = super(ChannelAttention, self).get_config()
        config.update({
            "filters": self.filters,
            "ratio": self.ratio,
        })
        return config

class SpatialAttention(Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = Conv2D(1, kernel_size, padding='same', activation='sigmoid',
                          kernel_initializer='he_normal', use_bias=False, name='sa_conv')

    def call(self, inputs):
        avg_pool = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True), name='sa_avg_pool')(inputs)
        max_pool = Lambda(lambda x: K.max(x, axis=-1, keepdims=True), name='sa_max_pool')(inputs)

        # Concatenate along the channel axis
        concat = Concatenate(axis=-1, name='sa_concat')([avg_pool, max_pool])
        attention = self.conv(concat)

        # Multiply attention scores with input feature map
        return Multiply(name='sa_multiply')([inputs, attention])

    def get_config(self):
        config = super(SpatialAttention, self).get_config()
        config.update({
            "kernel_size": self.kernel_size,
        })
        return config

def get_custom_objects():
    """Return a dictionary of custom objects needed for model loading."""
    return {
        'ChannelAttention': ChannelAttention,
        'SpatialAttention': SpatialAttention,
    }
