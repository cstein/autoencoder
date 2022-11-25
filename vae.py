import random

import tensorflow as tf
from tensorflow import keras
from tensorflow import dtypes


class Encoder(keras.layers.Layer):
    """ Encodes TO latent space """
    def __init__(self, unit_size, latent_size, batch_size):
        super(Encoder, self).__init__(name="encoder")
        self.unit_size = unit_size
        self.latent_size = latent_size
        self.batch_size = batch_size

        self.rnn = keras.layers.RNN(
                [keras.layers.LSTMCell(self.unit_size),
                 keras.layers.LSTMCell(self.unit_size),
                 keras.layers.LSTMCell(self.unit_size)],
                return_state=True
        )

    def build(self, input_shape):
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

    def call(self, inputs):
        # print("\n\n **** ENCODER ****")
        # print("    ... call ...")
        state = self.rnn(inputs)
        c, h = state[-1]
        mean = tf.matmul(h, self.w_mean) + self.b_mean
        sigma = tf.matmul(h, self.w_sigma) + self.b_sigma
        retval = mean + tf.exp(sigma / 2.0) + tf.random.normal(shape=[self.batch_size, self.latent_size], name="lol")
        loss = tf.reduce_mean(-0.5 * (1 + sigma - tf.square(mean) - tf.exp(sigma)))
        self.add_loss(loss)
        return retval


class Decoder(keras.layers.Layer):
    def __init__(self, unit_size, vocab_size, batch_size):
        super(Decoder, self).__init__(name="decoder")
        self.unit_size = unit_size
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.rnn = keras.layers.RNN(
                [keras.layers.LSTMCell(self.unit_size),
                 keras.layers.LSTMCell(self.unit_size),
                 keras.layers.LSTMCell(self.unit_size)],
                return_state=True,
                return_sequences=True,
        )

    def build(self, input_shape):
        self.rnn_initial_state = tuple(
                [
                    tf.compat.v1.nn.rnn_cell.LSTMStateTuple(
                            tf.zeros(shape=(self.batch_size, self.unit_size)),
                            tf.zeros(shape=(self.batch_size, self.unit_size))
                    ) for i in range(3)
                ]
        )

        w_init = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
        self.w = tf.Variable(
                initial_value=w_init(
                        shape=(self.unit_size, self.vocab_size),
                        dtype=dtypes.float32),
                name="weight"
        )

        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
                initial_value=b_init(shape=[self.vocab_size]),
                name="bias"
        )

    def call(self, inputs, *args, **kwargs):

        # print("\n\n **** DECODER ****")
        # print("    ... call ...")
        latent, x = inputs
        # print("  latent:", latent)
        # print("       x:", x)
        decoding_seq_length = tf.shape(x)[1]
        # change dimensions (no idea where data goes)
        # latent dimensions is [5, 200] and the expand_dims makes
        # a new dimension, i.e. [5, 200, ?]
        z = tf.tile(tf.expand_dims(latent, 1), [1, decoding_seq_length, 1])
        # print("        Z:", z)
        inp = tf.concat([z, x], axis=-1)
        # print("   INPUT:", inp)
        # print(" INITIAL STATE:", initial_state)
        y, a, b, c = self.rnn(inp, initial_state=self.rnn_initial_state)
        # print("  Decoder Y (shape):", y.shape)
        # print("          A (shape):", a.shape)
        # print("          B (shape):", b.shape)
        # print("          C (shape):", c.shape)

        # print("  Decoder S:", state)
        y = tf.reshape(y, shape=[self.batch_size * decoding_seq_length, -1])
        # print("  reshape:", y)
        # print("        w:", self.w)
        # print("        b:", self.b)
        y = tf.matmul(y, self.w) + self.b
        Y_logits = tf.reshape(y, shape=[self.batch_size, decoding_seq_length, -1])
        # print("   Y_logits:", Y_logits)
        y_out = tf.nn.softmax(Y_logits)
        # print("   Y       :", y_out)
        # Y_logits = tf.reshape(y, shape=[self.batch_size, decoding_seq_length])
        return y_out, Y_logits


class VariationalAutoEncoder(keras.Model):
    def __init__(self, latent_size, vocab_size, batch_size, unit_size, name="autoencoder"):
        super(VariationalAutoEncoder, self).__init__(name=name)
        self.batch_size = batch_size
        self.latent_size = latent_size
        self.encoder = Encoder(unit_size=unit_size, latent_size=latent_size, batch_size=batch_size)
        self.decoder = Decoder(unit_size=unit_size, vocab_size=vocab_size, batch_size=batch_size)

        embedding_encode = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
        self.embedding_encode = tf.Variable(
                initial_value=embedding_encode(shape=(latent_size, vocab_size)),
                dtype=dtypes.float32,
                trainable=True
        )

    def call(self, inputs):
        self.x = tf.nn.embedding_lookup(self.embedding_encode, tf.cast(inputs, dtype=dtypes.int32))

        latent = self.encoder(self.x)
        y, y_logits = self.decoder((latent, self.x))

        self.add_loss(self.encoder.losses)
        return tf.argmax(y, axis=2), y_logits

    @tf.function
    def sample(self):
        eps = tf.random.normal([self.batch_size, self.latent_size], mean=0.0, stddev=0.0)
        y, y_logits = self.decoder((eps, self.x))
        return tf.argmax(y, axis=2)
