""" Trains the variational autoencoder """
import argparse

import numpy as np
from rdkit import Chem
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import dtypes
import time

from vae import VariationalAutoEncoder


@tf.function
def train_step(inputs, outputs, lengths, autoencoder):
    with tf.GradientTape() as tape:
        reconstructed, y_log = autoencoder(inputs, training=True)

        weights = tf.sequence_mask(lengths, encoding_seq_length)
        weights = tf.cast(weights, dtype=dtypes.int32)
        weights = tf.cast(weights, dtype=dtypes.float32)
        seq_loss = tfa.seq2seq.sequence_loss(logits=y_log, targets=outputs, weights=weights)
        reconstr_loss = tf.reduce_mean(seq_loss)
        latent_loss = sum(autoencoder.losses)
        loss = reconstr_loss + latent_loss
    grads = tape.gradient(loss, autoencoder.trainable_weights)
    optimizer.apply_gradients(zip(grads, autoencoder.trainable_weights))
    return latent_loss, reconstr_loss


if __name__ == '__main__':
    import encoding

    ap = argparse.ArgumentParser("VAE", "python train.py my_smiles_file.smi", "Trains the variational autoencoder.")
    ap.add_argument("file", metavar="FILE", help="file with SMILES")
    ap.add_argument("-r", "--learning-rate", metavar="RATE", default=1.0e-3, type=float, dest="learning_rate",
                    help="Default is %(default)s.")
    ap.add_argument("--split", metavar="NUMBER", default=0.75, type=float, dest="split", help="fraction of data to use for training.")
    ap.add_argument("--max-data", metavar="NUMBER", default=100, type=int, dest="max_data", help="max subset of data.")
    ap.add_argument("-e", metavar="NUMBER", dest="epochs", default=10, type=int, help="Number of epochs. Default is %(default)s.")
    ap.add_argument("--batch-size", metavar="SIZE", default=10, type=int,
                    help="Size of the batch when training. Default is %(default)s.")

    encoder_ap = ap.add_argument_group("Encoder", "Controls options specific to the encoder.")
    encoder_ap.add_argument("--encoding-length",
                            metavar="LENGTH", default=110,
                            help="Length of encoding. Default is %(default)s.")
    encoder_ap.add_argument("--latent-length",
                            metavar="LENGTH", default=200,
                            help="Length of the latent space. Default is %(default)s.")

    decoder_ap = ap.add_argument_group("Decoder", "Controls options specific for the decoder.")
    rnn_ap = ap.add_argument_group("RNN Layers", "Controls options regarding how the Recurrent Neural Network is used.")
    rnn_ap.add_argument("--rnn-layers",
                        metavar="NUMBER", dest="rnn_layers_size", default=3, help="Number of layers in the RNN. Default is %(default)s.")
    rnn_ap.add_argument("--rnn-unit-size",
                        metavar="NUMBER", dest="rnn_dimensions", default=512, help="size of the RNN. Default is %(default)s.")

    args = ap.parse_args()
    print(args)

    smiles = encoding.load_smiles_file(args.file)
    can_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(s), True) for s in smiles]
    encoding_seq_length = args.encoding_length
    latent_size = args.latent_length
    batch_size = args.batch_size
    rnn_dimensions = args.rnn_dimensions
    rnn_layers_size = args.rnn_layers_size

    input, output, alphabet, vocabulary, lengths = encoding.encode(can_smiles, encoding_seq_length)

    # data preparation
    max_data = args.max_data
    inputs = input[0:max_data][:]
    lengths = lengths[0:max_data][:]
    n_split = int(args.split*len(inputs))

    train_input = inputs[0:n_split]
    train_output = output[0:n_split]
    test_input = inputs[n_split:max_data]

    train_lengths = lengths[0:n_split]
    test_lengths = lengths[n_split:max_data]

    v = VariationalAutoEncoder(latent_size=latent_size, vocab_size=len(alphabet), batch_size=batch_size,
                               rnn_num_dimensions=rnn_dimensions, rnn_num_layers=rnn_layers_size)

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    epochs = args.epochs
    lat_loss = 0.0
    rec_loss = 0.0
    for epoch in range(epochs):
        t_epoch = time.time()
        # print(f"Epoch = {epoch}")
        for step in range(len(train_input)//batch_size):
            t_train_step = time.time()
            n = np.random.randint(len(train_input), size=batch_size)
            train_repr = np.array([train_input[i] for i in n])
            train_repr_y = np.array([train_output[i] for i in n])
            l = np.array([train_lengths[i] for i in n])

            lat_loss, rec_loss = train_step(train_repr, train_repr_y, l, v)

            dt_step = time.time() - t_train_step
            if step % 20 == 0:
                print(f"   train step {step:4d}   lat loss {lat_loss:9.4f}   rec loss {rec_loss:9.4f}")

        # for step in range(len(test_input)//batch_size):
        #     t_train_step = time.time()
        #     n = np.random.randint(len(test_input), size=batch_size)
        #     test_repr = np.array([test_input[i] for i in n])
        #     l = np.array([test_lengths[i] for i in n])
        #
        #     reconstructed, y_log = v(test_repr, training=False)
        #
        #     weights = tf.sequence_mask(l, encoding_seq_length)
        #     weights = tf.cast(weights, dtype=dtypes.int32)
        #     weights = tf.cast(weights, dtype=dtypes.float32)
        #     loss = tf.reduce_mean(
        #             tfa.seq2seq.sequence_loss(logits=y_log, targets=test_repr, weights=weights)
        #     )
        #     dt_step = time.time() - t_train_step
        #     print(f"Test step = {step+1:02d}/{len(test_input) // batch_size:02d}. Time = {dt_step:.1f} sec. Loss = {loss_metric.result():.2f}")

        dt_epoch = time.time() - t_epoch
        print(f"Epoch {epoch+1:03d} finished in {dt_epoch/60:.1f} min. Lat. Loss = {lat_loss:7.4f} Ret. Loss = {rec_loss:7.4f}")

        v.save_weights("saved/checkpoint", save_format="tf")
        if (total_loss := lat_loss + rec_loss) < 0.4:
            break
        # tf.saved_model.save(v, "saved/checkpoint")
    print("done")
    v.summary()

    # for row in lol:
    #     print("  row:", np.sum(row.numpy()), "VAL", row.numpy())
    # tf.saved_model.save(v, "saved/checkpoint")
    # v.summary()
