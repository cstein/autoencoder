import tensorflow as tf
from tensorflow import dtypes
from tensorflow import keras


class Linear(keras.layers.Layer):
    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        # initialize weights to a normal distribution
        # but note that size is not given here
        w_init = tf.random_normal_initializer()

        # we initialize the weights to be randomly distributed
        # and of course they are trainable
        self.w = tf.Variable(
                initial_value=w_init(shape=(input_shape[-1], self.units), dtype=dtypes.float32),
                trainable=True
        )

        # the offset is also trainable but initialized to zero
        b_init = tf.zeros_initializer()
        self.offset = tf.Variable(
                initial_value=b_init(shape=(self.units, ), dtype=dtypes.float32),
                trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.offset


if __name__ == '__main__':
    x = tf.ones((2, 2))
    print(x)
    linear_model = Linear(4)
    y = linear_model(x)
    print(y)
    print(linear_model.weights)
