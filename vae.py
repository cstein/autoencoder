import random

import tensorflow as tf
from tensorflow import keras
from tensorflow import dtypes


class Encoder(keras.layers.Layer):
    """ Encoder in the variational autoencoder

        Encodes some input to the latent space.
    """
    def __init__(self,
                 unit_size: int,
                 latent_size: int,
                 batch_size: int,
                 mean: float,
                 stddev: float):
        """ Initializes the Encoder

            :param unit_size: the size of each of the RNNs
            :param latent_size: the size of the latent space
            :param batch_size: size of the incoming batches of data
            :param mean: mean added to the latent space vector
            :param stddev: spread of the mean added to the latent space vector
        """
        super(Encoder, self).__init__(name="encoder")
        self.unit_size = unit_size
        self.latent_size = latent_size
        self.batch_size = batch_size
        self.mean = mean
        self.standard_deviation = stddev

        self.rnn = keras.layers.RNN(
                [keras.layers.LSTMCell(self.unit_size),
                 keras.layers.LSTMCell(self.unit_size),
                 keras.layers.LSTMCell(self.unit_size)],
                return_state=True
        )

    def build(self, input_shape):
        """ Builds the neural network for the encoder """
        w_init = keras.initializers.GlorotUniform(seed=random.randint(0, 65536))
        self.w_mean = tf.Variable(
                initial_value=w_init(shape=[self.unit_size, self.latent_size]),
                name="mean_weight"
        )
        b_init = tf.zeros_initializer()
        self.b_mean = tf.Variable(
                initial_value=b_init(shape=[self.latent_size]),
                name="mean_bias"
        )

        self.w_sigma = tf.Variable(
                initial_value=w_init(shape=[self.unit_size, self.latent_size]),
                name="sigma_weight"
        )

        self.b_sigma = tf.Variable(
                initial_value=b_init(shape=[self.latent_size]),
                name="sigma_bias"
        )

    def call(self, inputs, *args, **kwargs):
        """ Executes the Encoder and computes the latent space vector """
        state = self.rnn(inputs)
        c, h = state[-1]
        mean = tf.matmul(h, self.w_mean) + self.b_mean
        sigma = tf.matmul(h, self.w_sigma) + self.b_sigma
        retval = mean + tf.exp(sigma / 2.0) * tf.random.normal(shape=[self.batch_size, self.latent_size], mean=self.mean, stddev=self.standard_deviation)
        loss = tf.reduce_mean(-0.5 * (1 + sigma - tf.square(mean) - tf.exp(sigma)))
        self.add_loss(loss)
        return retval


class Decoder(keras.layers.Layer):
    def __init__(self, unit_size, vocab_size, batch_size):
        """ Initializes the decoder

            :param unit_size: the size of each of the RNNs
            :param vocab_size: the number of characters to remember
            :param batch_size: size of the incoming batches of data

        """
        super(Decoder, self).__init__(name="decoder")
        self.unit_size = unit_size
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.rnn = keras.layers.RNN(
                [keras.layers.LSTMCell(self.unit_size),
                 keras.layers.LSTMCell(self.unit_size),
                 keras.layers.LSTMCell(self.unit_size)],
                return_state=True,
                return_sequences=True
        )

    def build(self, input_shape):
        w_init = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
        self.w = tf.Variable(
                initial_value=w_init(
                        shape=[self.unit_size, self.vocab_size],
                        dtype=dtypes.float32),
                name="weight"
        )

        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
                initial_value=b_init(shape=[self.vocab_size]),
                name="bias"
        )

    def call(self, inputs, *args, **kwargs):
        input_state = kwargs.get("state")
        # we only unpack the first (y) value (the predicted sequence) and
        # store the rest as the state using extended unpacking (the * operator)
        y, *updated_state = self.rnn(inputs, initial_state=input_state)

        seq_length = tf.shape(y)[1]
        y = tf.reshape(y, shape=[self.batch_size * seq_length, -1])
        y = tf.matmul(y, self.w) + self.b
        logits = tf.reshape(y, shape=[self.batch_size, seq_length, -1])
        return tf.nn.softmax(logits), logits, updated_state


class VariationalAutoEncoder(keras.Model):
    def __init__(self,
                 latent_size,
                 vocab_size,
                 batch_size,
                 unit_size,
                 mean=0.0,
                 stddev=1.0,
                 name="autoencoder"):
        """ Creates the variational autoencoder

            :param latent_size: the size of the latent space
            :param vocab_size: the number of characters to remember
            :param batch_size: size of the incoming batches of data
            :param unit_size: the size of each of the RNNs
            :param mean: mean added to the latent space vector
            :param stddev: spread of the mean added to the latent space vector
            :param name: the internal name
        """
        super(VariationalAutoEncoder, self).__init__(name=name)
        self.batch_size = batch_size
        self.latent_size = latent_size
        self.mean = mean
        self.standard_deviation = stddev

        # we need to be able to reset the RNN when we
        # want to make a prediction, so we need to store
        # the initial empty state
        self.rnn_initial_state = tuple(
                [
                    tf.compat.v1.nn.rnn_cell.LSTMStateTuple(
                            tf.zeros(shape=(batch_size, unit_size)),
                            tf.zeros(shape=(batch_size, unit_size))
                    ) for i in range(3)
                ]
        )
        self.rnn_state = self.rnn_initial_state[:]

        embedding_encode = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
        self.embedding_encode = tf.Variable(
                initial_value=embedding_encode(shape=(latent_size, vocab_size)),
                dtype=dtypes.float32,
                trainable=True
        )

        self.encoder = Encoder(unit_size=unit_size, latent_size=latent_size, batch_size=batch_size, mean=mean, stddev=stddev)
        self.decoder = Decoder(unit_size=unit_size, vocab_size=vocab_size, batch_size=batch_size)

    def call(self, inputs, training: bool = False, mask: bool = None):
        x = tf.nn.embedding_lookup(self.embedding_encode, tf.cast(inputs, dtype=dtypes.int32))
        if training:
            latent = self.encoder(x)
        else:
            latent = tf.random.normal([self.batch_size, self.latent_size], mean=self.mean, stddev=self.standard_deviation)
        z = tf.tile(tf.expand_dims(latent, 1), [1, tf.shape(x)[1], 1])
        inp = tf.concat([z, x], axis=-1)

        y, y_logits, state = self.decoder(inp, state=self.rnn_state)
        if not training:
            self.rnn_state = state[:]

        self.add_loss(self.encoder.losses)
        return tf.argmax(y, axis=2), y_logits

    def reset(self):
        """ Reset the network to its initial state

            This is used when making predictions
        """
        self.rnn_state = self.rnn_initial_state[:]

