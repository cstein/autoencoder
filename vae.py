import random

import tensorflow as tf
from tensorflow import keras
from tensorflow import dtypes


class Encoder(keras.layers.Layer):
    """ Encoder in the variational autoencoder

        Encodes some input to the latent space.
    """
    def __init__(self,
                 batch_size: int,
                 latent_size: int,
                 rnn_num_dimensions: int,
                 rnn_num_layers: int,
                 mean: float,
                 stddev: float):
        """ Initializes the Encoder
            :param batch_size: size of the incoming batches of data
            :param latent_size: the size of the latent space
            :param rnn_num_dimensions: the size of each of the RNNs
            :param rnn_num_layers: number of layers in the rnn
            :param mean: mean added to the latent space vector
            :param stddev: spread of the mean added to the latent space vector
        """
        super(Encoder, self).__init__(name="encoder")
        self.rnn_num_dimensions = rnn_num_dimensions
        self.latent_size = latent_size
        self.batch_size = batch_size
        self.mean = mean
        self.standard_deviation = stddev

        self.rnn = keras.layers.RNN(
                [keras.layers.LSTMCell(rnn_num_dimensions) for _ in range(rnn_num_layers)],
                return_state=True
        )

    def build(self, input_shape):
        """ Builds the neural network for the encoder """
        w_init = keras.initializers.GlorotUniform(seed=random.randint(0, 65536))
        self.w_mean = tf.Variable(
                initial_value=w_init(shape=[self.rnn_num_dimensions, self.latent_size]),
                name="mean_weight"
        )
        b_init = tf.zeros_initializer()
        self.b_mean = tf.Variable(
                initial_value=b_init(shape=[self.latent_size]),
                name="mean_bias"
        )

        self.w_sigma = tf.Variable(
                initial_value=w_init(shape=[self.rnn_num_dimensions, self.latent_size]),
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
    def __init__(self,
                 batch_size: int,
                 vocab_size: int,
                 rnn_num_dimensions: int,
                 rnn_num_layers: int):
        """ Initializes the decoder
            :param vocab_size: the number of characters to remember
            :param batch_size: size of the incoming batches of data
            :param rnn_num_layers: number of layers in the rnn
            :param rnn_num_dimensions: the size of each of the RNNs
        """
        super(Decoder, self).__init__(name="decoder")
        self.rnn_num_dimensions = rnn_num_dimensions
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.rnn = keras.layers.RNN(
                [keras.layers.LSTMCell(rnn_num_dimensions) for _ in range(rnn_num_layers)],
                return_state=True,
                return_sequences=True
        )

    def build(self, input_shape):
        w_init = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
        self.w = tf.Variable(
                initial_value=w_init(
                        shape=[self.rnn_num_dimensions, self.vocab_size],
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
                 latent_size: int,
                 vocab_size: int,
                 batch_size: int,
                 rnn_num_dimensions: int,
                 rnn_num_layers: int,
                 mean: float = 0.0,
                 stddev: float = 1.0,
                 name="autoencoder"):
        """ Creates the variational autoencoder
            :param latent_size: the size of the latent space
            :param vocab_size: the number of characters to remember
            :param batch_size: size of the incoming batches of data
            :param rnn_num_dimensions: the size of each of the RNNs
            :param rnn_num_layers: number of layers in the rnn
            :param mean: mean added to the latent space vector
            :param stddev: spread of the mean added to the latent space vector
            :param name: the internal name
        """
        super(VariationalAutoEncoder, self).__init__(name=name)
        self.batch_size = batch_size
        self.latent_size = latent_size
        self.mean = mean
        self.stddev = stddev

        # we need to be able to reset the RNNs when we
        # want to make a prediction, so we need to store
        # an initial empty state that we can reuse. See
        # the method `reset` below
        self.rnn_initial_state = tuple(
                [
                    tf.compat.v1.nn.rnn_cell.LSTMStateTuple(
                            tf.zeros(shape=(batch_size, rnn_num_dimensions)),
                            tf.zeros(shape=(batch_size, rnn_num_dimensions))
                    ) for _ in range(rnn_num_layers)
                ]
        )
        self.rnn_state = self.rnn_initial_state[:]

        embedding_encode = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
        self.embedding_encode = tf.Variable(
                initial_value=embedding_encode(shape=(latent_size, vocab_size)),
                dtype=dtypes.float32,
                trainable=True
        )

        self.encoder = Encoder(batch_size=batch_size, latent_size=latent_size,
                               rnn_num_dimensions=rnn_num_dimensions, rnn_num_layers=rnn_num_layers,
                               mean=mean, stddev=stddev)
        self.decoder = Decoder(batch_size=batch_size, vocab_size=vocab_size,
                               rnn_num_dimensions=rnn_num_dimensions, rnn_num_layers=rnn_num_layers)

    def call(self, inputs, training: bool = False, mask: bool = None):
        (inps, properties) = inputs

        x = tf.nn.embedding_lookup(self.embedding_encode, tf.cast(inps, dtype=dtypes.int32))
        properties = tf.tile(tf.expand_dims(properties, 1), [1, tf.shape(x)[1], 1])

        if training:
            encoder_input = tf.concat([x, properties], axis=-1)
            latent = self.encoder(encoder_input)
            self.add_loss(self.encoder.losses)
        else:
            latent = tf.random.normal([self.batch_size, self.latent_size], mean=self.mean, stddev=self.stddev)
        padded_latent = tf.tile(tf.expand_dims(latent, 1), [1, tf.shape(x)[1], 1])
        decoder_input = tf.concat([padded_latent, x, properties], axis=-1)

        y, y_logits, state = self.decoder(decoder_input, state=self.rnn_state)
        if not training:
            self.rnn_state = state[:]

        return tf.argmax(y, axis=2), y_logits

    def reset(self):
        """ Reset the network to its initial state

            This is used when making predictions
        """
        self.rnn_state = self.rnn_initial_state[:]
