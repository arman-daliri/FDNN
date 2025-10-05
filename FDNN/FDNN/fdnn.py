import tensorflow as tf
from tensorflow.keras import layers, models


class MembershipFunctionLayer(tf.keras.layers.Layer):
    """
    A custom fuzzy membership function layer for fuzzy deep neural networks.
    """

    def __init__(self, units):
        super(MembershipFunctionLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.mu = self.add_weight(
            shape=(self.units, input_shape[-1]),
            initializer="random_normal",
            trainable=True
        )
        self.sigma = self.add_weight(
            shape=(self.units, input_shape[-1]),
            initializer="ones",
            trainable=True
        )

    def call(self, inputs):
        x = tf.expand_dims(inputs, axis=1)
        mu = tf.expand_dims(self.mu, axis=0)
        sigma = tf.expand_dims(self.sigma, axis=0)
        return tf.exp(-tf.square((x - mu) / sigma))


class Reduce_Prod_Layer(tf.keras.Layer):
    def call(self, membership, axis):
        return tf.reduce_prod(membership, axis=axis)


class Concat_Layer(tf.keras.Layer):
    def call(self, l, axis):
        return tf.concat(l, axis=axis)


def build_fdnn(
        input_dim,
        membership_units=3,
        dense_units=128,
        dropout_rate=0.4):
    """
    Builds and compiles a fuzzy deep neural network (FDNN) model.
    """
    inputs = tf.keras.Input(shape=(input_dim,))
    membership = MembershipFunctionLayer(units=membership_units)(inputs)
    fuzzy_rule = Reduce_Prod_Layer()(membership, axis=-1)
    fused = Concat_Layer()([inputs, fuzzy_rule], axis=-1)
    dense = layers.Dense(dense_units, activation='relu')(fused)
    dropout = layers.Dropout(dropout_rate)(dense)
    output = layers.Dense(2, activation='softmax')(dropout)

    model = models.Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
