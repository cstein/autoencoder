import argparse

from rdkit import Chem
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import dtypes
import numpy as np
import time

from vae import VariationalAutoEncoder

if __name__ == '__main__':
    import encoding

    ap = argparse.ArgumentParser("VAE", "python vae.py my_smiles_file.smi", "Trains the variational autoencoder")
    ap.add_argument("file", metavar="FILE", help="file with SMILES")
    ap.add_argument("-r", "--learning-rate", metavar="RATE", default=1.0e-3, type=float, dest="learning_rate",
                    help="Default is %(default)s.")
    ap.add_argument("--split", metavar="NUMBER", default=0.75, type=float, dest="split", help="fraction of data to use for training.")
    ap.add_argument("--max-data", metavar="NUMBER", default=100, type=int, dest="max_data", help="max subset of data.")
    ap.add_argument("-e", metavar="NUMBER", dest="epochs", default=10, type=int, help="Number of epochs. Default is %(default)s.")

    encoder_ap = ap.add_argument_group("Encoder")
    encoder_ap.add_argument("--encoding-length", metavar="LENGTH", default=110,
                    help="Length of encoding. Default is %(default)s.")
    encoder_ap.add_argument("--latent-length", metavar="LENGTH", default=200,
                    help="Length of the latent space. Default is %(default)s.")

    decoder_ap = ap.add_argument_group("Decoder")
    ap.add_argument("--batch-size", metavar="SIZE", default=10, help="Size of the batch when training. Default is %(default)s.")
    rnn_ap = ap.add_argument_group("RNN Layers", "Controls options regarding how the RNN is used.")
    rnn_ap.add_argument("--rnn-layers", metavar="NUMBER", dest="rnn_layers_size", default=3, help="Number of layers in the RNN. Default is %(default)s.")
    rnn_ap.add_argument("--rnn-unit-size", metavar="NUMBER", dest="rnn_unit_size", default=512, help="size of the RNN. Default is %(default)s.")

    args = ap.parse_args()
    smiles = encoding.load_smiles_file(args.file)
    can_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(s), True) for s in smiles]
    print("--- Settings ---")
    encoding_seq_length = args.encoding_length
    latent_size = args.latent_length
    batch_size = args.batch_size
    unit_size = args.rnn_unit_size
    print("encoding_seq_length", encoding_seq_length)
    print("latent_size", latent_size)
    print("batch_size", batch_size)
    input, output, alphabet, vocabulary, lengths = encoding.encode(can_smiles, encoding_seq_length)
    print("------")
    print("input shape:", np.shape(input))
    print("alphabet   :", np.shape(alphabet))
    print("lengths    :", np.shape(lengths))
    print("------")

    # data preparation
    max_data = args.max_data
    input = input[0:max_data][:]
    lengths = lengths[0:max_data][:]
    n_split = int(args.split*len(input))

    train_input = input[0:n_split]
    test_input = input[n_split:max_data]

    train_lengths = lengths[0:n_split]
    test_lengths = lengths[n_split:max_data]
    # done

    v = VariationalAutoEncoder(latent_size=latent_size, vocab_size=len(alphabet), batch_size=batch_size,
                               unit_size=unit_size)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_metric = tf.keras.metrics.Mean()

    epochs = args.epochs
    for epoch in range(epochs):
        t_epoch = time.time()
        print(f"Epoch = {epoch}")
        for step in range(len(train_input)//batch_size):
            t_train_step = time.time()
            n = np.random.randint(len(train_input), size=batch_size)
            train_repr = np.array([train_input[i] for i in n])
            l = np.array([train_lengths[i] for i in n])
            with tf.GradientTape() as tape:
                reconstructed, y_log = v(train_repr)

                weights = tf.sequence_mask(l, encoding_seq_length)
                weights = tf.cast(weights, dtype=dtypes.int32)
                weights = tf.cast(weights, dtype=dtypes.float32)
                seq_loss = tfa.seq2seq.sequence_loss(logits=y_log, targets=train_repr, weights=weights)
                loss = tf.reduce_mean(
                        seq_loss
                )
                loss += sum(v.losses)

            loss_metric(loss)
            optimizer.minimize(loss, v.trainable_weights, tape=tape)

            dt_step = time.time() - t_train_step
            print(f"Train step = {step+1:02d}/{len(train_input)//batch_size:02d}. Time = {dt_step:.1f} sec. Loss = {loss_metric.result():.2f}")

        for step in range(len(test_input)//batch_size):
            t_train_step = time.time()
            n = np.random.randint(len(test_input), size=batch_size)
            test_repr = np.array([test_input[i] for i in n])
            l = np.array([test_lengths[i] for i in n])

            reconstructed, y_log = v(test_repr)

            weights = tf.sequence_mask(l, encoding_seq_length)
            weights = tf.cast(weights, dtype=dtypes.int32)
            weights = tf.cast(weights, dtype=dtypes.float32)
            loss = tf.reduce_mean(
                    tfa.seq2seq.sequence_loss(logits=y_log, targets=test_repr, weights=weights)
            )
            loss_metric(loss)
            dt_step = time.time() - t_train_step
            print(f"Test step = {step+1:02d}/{len(test_input) // batch_size:02d}. Time = {dt_step:.1f} sec. Loss = {loss_metric.result():.2f}")

        dt_epoch = time.time() - t_epoch
        # v.save_weights("saved/checkpoint", save_format="tf")
        # v.save("saved/checkpoint.tf", save_format="tf")
        print(f"Epoch {epoch+1:03d} finished in {dt_epoch/60:.1f} min.")
    # v.sample()
    tf.saved_model.save(v, "saved/checkpoint")
    v.summary()
