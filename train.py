""" Trains the variational autoencoder """
import argparse
import json
import time

import numpy as np
import pandas as pd
from rdkit import Chem
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import dtypes

import encoding
from vae import VariationalAutoEncoder
from utils.state import save

has_selfies = True
try:
    import selfies as sf
except ImportError:
    print("selfies not found.")
    has_selfies = False

"""
   Comments from Lars:
      * In general, smaller (rather than larger) batch sizes are wanted because of the need
        to avoid local minima. The smaller batch sizes allows us to "jump" around and avoid
        these local minima.
        
        The general idea is to have 100 % utilization on the GPU with the *smallest* possible
        batch size. So in practice, we aim for large enough batch-sizes so the GPU does not
        crash from lack of memory
"""


@tf.function
def train_step(inputs: np.ndarray,
               outputs: np.ndarray,
               lengths: np.ndarray,
               encoding_length: int,
               properties: np.ndarray,
               optimizer: tf.keras.optimizers.Adam,
               autoencoder: VariationalAutoEncoder) -> tuple[float, float]:
    """ Performs a single training step (forward pass and backward pass)

        :param inputs: encoded inputs used for training
        :param outputs: encoded target outputs used for training
        :param lengths: the lengths of the target outputs
        :param encoding_length: the overall length of encoding
        :param properties: the properties to use in training
        :param optimizer: the optimizer to use during training
        :param autoencoder: the autoencoder to use
    """
    with tf.GradientTape() as tape:
        latent_loss, reconstr_loss = calculate_loss(inputs, outputs, lengths, encoding_length, properties, autoencoder)
        loss = latent_loss + reconstr_loss
    grads = tape.gradient(loss, autoencoder.trainable_weights)
    optimizer.apply_gradients(zip(grads, autoencoder.trainable_weights))
    return latent_loss, reconstr_loss

@tf.function
def calculate_loss(inputs: np.ndarray,
                   outputs: np.ndarray,
                   lengths: np.ndarray,
                   encoding_length: int,
                   properties: np.ndarray,
                   autoencoder: VariationalAutoEncoder) -> tuple[float, float]:
    """ Calculates losses """
    mask = tf.sequence_mask(lengths, encoding_length)
    _, y_log = autoencoder((inputs, properties), training=True, mask=mask)

    weights = tf.cast(mask, dtype=dtypes.int32)
    weights = tf.cast(weights, dtype=dtypes.float32)
    seq_loss = tfa.seq2seq.sequence_loss(logits=y_log, targets=outputs, weights=weights)
    reconstr_loss = tf.reduce_mean(seq_loss)
    latent_loss = sum(autoencoder.losses)
    return latent_loss, reconstr_loss


def is_positive_number(value):
    value_as_int = int(value)
    if value_as_int < 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer.")
    return value_as_int


if __name__ == '__main__':
    learning_rate = 1.0e-3
    learning_convergence = 0.4
    latent_length = 200

    ap = argparse.ArgumentParser("VAE", "python train.py my_smiles_file.smi", "Trains the variational autoencoder and stores the model.")
    ap.add_argument("file", metavar="FILE", help=".csv file with a minimum of one column named SMILES with rows of smiles strings. Optional columns with molecular data.")
    ap.add_argument("--config", metavar="FILE", default="train.json", type=str, dest="config_file",
                    help="Configuration file to store model and training parameters. Use to reproduce or restart training or when sampling to correctly restore a model. Default is %(default)s.")
    ap.add_argument("-r", "--learning-rate", metavar="RATE", default=learning_rate, type=float, dest="learning_rate",
                    help="Default is %(default)s.")
    ap.add_argument("-c", "--convergence", metavar="NUMBER", default=learning_convergence, type=float, dest="learning_convergence",
                    help="Stop training when the combined loss (reconstruction + latent) falls below this value. Default is %(default)s.")
    ap.add_argument("--split", metavar="NUMBER", default=0.75, type=float, dest="split", help="fraction of data to use for training.")
    ap.add_argument("--max-data", metavar="NUMBER", default=100, type=int, dest="max_data", help="Max subset of input data to use. Default is %(default)s.")
    ap.add_argument("-e", metavar="NUMBER", dest="epochs", default=10, type=int, help="Number of epochs. Default is %(default)s.")
    ap.add_argument("--batch-size", metavar="SIZE", default=10, type=int,
                    help="Size of the batch when training. Default is %(default)s.")
    ap.add_argument("-p", "--properties", nargs="*", default=None, dest="properties", metavar="NAME", help="Colum names in the input .csv file to use for additional properties of the VAE.")

    encoder_ap = ap.add_argument_group("Encoder", "Controls options specific to the encoder.")
    encoder_ap.add_argument("--encoding-length",
                            metavar="SIZE", default=110,
                            help="Length of encoding. Default is %(default)s.")
    encoder_ap.add_argument("--latent-length",
                            metavar="SIZE", default=latent_length,
                            help="Length of the latent space. Default is %(default)s.")

    decoder_ap = ap.add_argument_group("Decoder", "Controls options specific for the decoder.")

    rnn_ap = ap.add_argument_group("RNN Layers", "Controls options regarding how the Recurrent Neural Network is used.")
    rnn_ap.add_argument("--rnn-layers",
                        metavar="NUMBER", dest="rnn_layers_size", default=3, help="Number of layers in the RNN. Default is %(default)s.")
    rnn_ap.add_argument("--rnn-size",
                        metavar="NUMBER", dest="rnn_dimensions", default=512, help="Size of each unit in the RNN. Default is %(default)s.")

    backup_ap = ap.add_argument_group("Backup", "Controls options specific to how and when backups are created during training.")
    backup_ap.add_argument("--backup-folder", metavar="FOLDER", default="saved", type=str, dest="backup_folder",
                           help="Folder to store backup in")
    backup_ap.add_argument("--backup-checkpoint", metavar="FILE", default="checkpoint", type=str,
                           dest="backup_checkpoint",
                           help="Name of the backup")
    backup_ap.add_argument("--backup-numbering", action="store_true", default=False, dest="backup_numbering",
                           help="Specify this argument to store models with the epoch number. Rate of saving is specified in --backup-rate.")
    backup_ap.add_argument("--backup-rate", metavar="NUMBER", default=0, type=is_positive_number, dest="backup_rate",
                           help="Store a backup every --backup-rate epochs. The last training step will always be saved. Default is 0 which means that we do not save backups during training.")

    args = ap.parse_args()
    with open(args.config_file, "w") as f:
        json.dump(vars(args), f, indent=4)
    print(args)

    encoding_seq_length = args.encoding_length
    latent_length = args.latent_length
    batch_size = args.batch_size
    rnn_dimensions = args.rnn_dimensions
    rnn_layers_size = args.rnn_layers_size
    backup_folder = args.backup_folder
    backup_checkpoint = args.backup_checkpoint
    backup_rate = args.backup_rate
    learning_rate = args.learning_rate
    learning_convergence = args.learning_convergence

    if args.backup_numbering:
        if backup_rate == 0:
            print("Backup with numbers are specified but --backup-rate not specified. Setting --backup-rate to 1.")
            backup_rate = 1

    try:
        df: pd.DataFrame = encoding.load_csv_file(args.file)
    except ValueError:
        raise

    smiles = df["SMILES"].values
    can_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(s), True) for s in smiles]
    selfies = []
    if has_selfies:
        selfies = [sf.encoder(s) for s in can_smiles]

    p = [[] for _ in can_smiles]
    if args.properties is not None:
        p = np.array(
                [df[value].values for value in args.properties]
        ).transpose()

    # encoding of input to one-hot encoding
    input, output, alphabet, vocabulary, lengths = encoding.encode(can_smiles, encoding_seq_length)
    if has_selfies:
        input, output, alphabet, vocabulary, lengths = encoding.encode_selfies(selfies, encoding_seq_length)

    # data preparation for training and testing
    max_data = args.max_data
    inputs = input[0:max_data][:]
    outputs = output[0:max_data][:]
    lengths = lengths[0:max_data][:]
    n_split = int(args.split*len(inputs))

    # now we split up the data in train and test
    train_input = inputs[0:n_split]
    train_output = outputs[0:n_split]
    train_lengths = lengths[0:n_split]
    train_properties = p[0:n_split]
    test_input = inputs[n_split:max_data]
    test_output = outputs[n_split:max_data]
    test_lengths = lengths[n_split:max_data]
    test_properties = p[n_split:max_data]

    v = VariationalAutoEncoder(latent_size=latent_length, vocab_size=len(vocabulary), batch_size=batch_size,
                               rnn_num_dimensions=rnn_dimensions, rnn_num_layers=rnn_layers_size,
                               encoding_length=encoding_seq_length)

    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    epochs = args.epochs
    lat_loss = 0.0
    rec_loss = 0.0
    for epoch in range(epochs):
        t_epoch = time.time()
        train_loss = 0.0
        test_loss = 0.0
        for step in range(len(train_input)//batch_size):
            t_train_step = time.time()

            n = np.random.randint(len(train_input), size=batch_size)
            train_repr_x = np.array([train_input[i] for i in n])
            train_repr_y = np.array([train_output[i] for i in n])
            weights = np.array([train_lengths[i] for i in n])
            properties = np.array([train_properties[i] for i in n])

            train_lat_loss, train_rec_loss = train_step(train_repr_x, train_repr_y, weights, encoding_seq_length, properties, adam_optimizer, v)
            train_loss = train_lat_loss + train_rec_loss

            dt_step = time.time() - t_train_step

            # lat_loss, rec_loss = calculate_loss(test)
            # if step % 50 == 0:
            #    print(f"   train step {step:4d}   lat loss {lat_loss:9.4f}   rec loss {rec_loss:9.4f}")

        for step in range(len(test_input)//batch_size):
            t_train_step = time.time()
            n = np.random.randint(len(test_input), size=batch_size)
            test_repr_x = np.array([test_input[i] for i in n])
            test_repr_y = np.array([test_output[i] for i in n])
            weights = np.array([test_lengths[i] for i in n])
            properties = np.array([test_properties[i] for i in n])
            test_rec_loss, test_lat_loss = calculate_loss(test_repr_x, test_repr_y, weights, encoding_seq_length, properties, v)
            test_loss = test_lat_loss + test_rec_loss

        dt_epoch = time.time() - t_epoch
        print(f"Epoch {epoch+1:03d} finished in {dt_epoch/60:.1f} min. loss(train, test) = ({train_loss:6.4f} {test_loss:7.4f})")

        # here we handle backup
        if args.backup_numbering:
            backup_checkpoint = f"{args.backup_checkpoint}_{epoch+1:03d}.ckpt"
        if backup_rate > 0:
            if (epoch+1) % backup_rate == 0:
                v.save_weights(f"{backup_folder}/{backup_checkpoint}", save_format="tf")

        # we break of the learning is sufficient
        if (total_loss := train_loss) < learning_convergence:
            break

    # v.save_weights(f"{backup_folder}/{backup_checkpoint}", save_format="tf")
    save_directory = save(v, f"{backup_folder}/{backup_checkpoint}")
    print(f"backup stored in {save_directory}")
    print("done")
    v.summary()
