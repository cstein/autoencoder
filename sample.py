import argparse
import json
import os

import pandas as pd
import numpy as np
from rdkit import Chem
import tensorflow as tf

from vae import VariationalAutoEncoder
from utils.state import restore
import encoding

has_selfies = True
try:
    import selfies as sf
except ImportError:
    print("selfies not found.")
    has_selfies = False


def generate_encoded_result(autoencoder: VariationalAutoEncoder, encoding_length: int, start_codon, properties) -> np.ndarray:
    vae_results = []
    generated_vector = start_codon[:]

    # we reset the internal state of the
    # autoencoder for every result we generate
    autoencoder.reset()
    for j in range(encoding_length):
        generated_vector, _ = autoencoder((generated_vector, properties))
        vae_results.append(generated_vector)

    return np.asarray(np.concatenate(vae_results, 1).astype(int).squeeze())


def generate_random_vector(batch_size: int, latent_size: int, mean: float, stddev: float):
    return tf.random.normal([batch_size, latent_size], mean=mean, stddev=stddev)


def convert_to_smiles(vector: np.ndarray, char):
    list_char = list(char)
    vector = vector.astype(int)
    return "".join(map(lambda x: list_char[x], vector)).strip()


if __name__ == '__main__':
    ap = argparse.ArgumentParser("Sampler", "python sample.py train.json", "samples from a VAE")
    ap.add_argument("-i", "--iterations", metavar="NUMBER", type=int, default=10, dest="num_iterations", help="number of iterations to run. Default is %(default)s.")
    ap.add_argument("-f", dest="file", metavar="FILE", help=".json file with configuration.")
    ap.add_argument("-o", "--output", metavar="FILE", default=None, help="The filename (.smi format) to use for output.")
    ap.add_argument("-q", action="store_false", default=True, dest="is_verbose", help="specify to hide output.")
    ap.add_argument("-b", "--from-backup", metavar="FILE", default=None, help="the backup file to use")
    ap.add_argument("-p", "--properties", nargs="*", type=float, dest="properties")
    latent_ap = ap.add_argument_group("Latent Sampling")
    latent_ap.add_argument("--mean", metavar="NUMBER", default=0.0, type=float,
                           dest="mean", help="mean value to adjust latent space sampling")
    latent_ap.add_argument("--sigma", metavar="NUMBER", default=0.2, type=float,
                           dest="sigma", help="standard deviation of latent space sampling")
    latent_ap.add_argument("-s", dest="smiles", default=None, type=str,
                           help="SMILES string to build latent space from.")


    args = ap.parse_args()
    print(args)
    with open(args.file, "r") as f:
        settings = json.load(f)

    latent_size = settings["latent_length"]
    batch_size = settings["batch_size"]
    unit_size = settings["rnn_dimensions"]
    rnn_num_layers = settings["rnn_layers_size"]
    encoding_seq_length = settings["encoding_length"]
    smiles_file = settings["file"]
    num_epochs = settings["epochs"]
    backup_folder = settings["backup_folder"]
    backup_checkpoint = settings["backup_checkpoint"]
    backup_numbering = settings["backup_numbering"]
    model_path = os.path.join(backup_folder, backup_checkpoint)
    full_backup_path = os.path.join(backup_folder, backup_checkpoint)
    if backup_numbering:
        filename, _ = os.path.splitext(backup_checkpoint)
        # we iterate from the last iteration in an attempt to find
        # the file with the _highest_ epoch in the filename
        for i in range(num_epochs, 0, -1):
            potential_model_path = os.path.join(backup_folder, f"{filename}_{i:03d}.ckpt-1")
            # we check if the .index file of the model is present
            # which signals that this is the model to use.
            if os.path.isfile(potential_model_path + ".index"):
                model_path = potential_model_path
                break

    model_properties = settings["properties"]
    if model_properties is None and args.properties is not None:
        print(f"Error: model was not trained with properties but you requested {len(args.properties)} with values:", args.properties)
        raise ValueError("Error: model is trained without properties.")
    elif model_properties is not None and args.properties is None:
        print(f"Error: model was trained with {len(model_properties)} properties but none was provided as argument -p.")
        raise ValueError("Error: no properties provided on command line.")
    elif model_properties is not None and args.properties is not None:
        print(" * PROPERTIES")
        if len(model_properties) != len(args.properties):
            for key in model_properties:
                print(f"   {key}")
            print(f"Error: number of model properties: {len(model_properties)}, does not match the requested number of properties: {len(args.properties)}.")
            raise ValueError("Error: property mismatch between model and request")
        else:
            for key, v in zip(model_properties, args.properties):
                print(f"   {key}: {v}")
    else:
        pass

    mean = args.mean
    stddev = args.sigma

    try:
        df: pd.DataFrame = encoding.load_csv_file(smiles_file)
    except ValueError:
        raise

    input_smiles = df["SMILES"].values
    can_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(s), True) for s in input_smiles]
    predefined_input, output, alphabet, vocabulary, lengths = encoding.encode(can_smiles, encoding_seq_length)
    selfies = []
    if has_selfies:
        selfies = [sf.encoder(s) for s in can_smiles]
        predefined_input, output, alphabet, vocabulary, lengths = encoding.encode_selfies(selfies, encoding_seq_length)
    print("selfies:", has_selfies)

    start_codon = np.array([np.array(list(map(vocabulary.get, 'X'))) for _ in range(batch_size)])
    properties = [[] for _ in range(batch_size)]
    if args.properties is not None:
        properties = np.array([args.properties for _ in range(batch_size)])

    vocab_size = len(vocabulary)
    outputs = []
    output_smiles = []
    v = VariationalAutoEncoder(latent_size=latent_size, vocab_size=vocab_size, batch_size=batch_size,
                               rnn_num_dimensions=unit_size, rnn_num_layers=rnn_num_layers,
                               encoding_length=encoding_seq_length,
                               mean=mean, stddev=stddev)

    print("... attempts to restore model from path:", model_path)
    restore(v, args.from_backup)
    # latent = tf.random.normal([batch_size, latent_size], mean=mean, stddev=self.stddev)
    # autoencoder.set_latent_vector()

    for i in range(args.num_iterations):
        # in each iteration, we generate a new random latent vector
        if args.smiles is None:
            v.set_latent_vector(
                    generate_random_vector(batch_size, latent_size, mean, stddev)
            )
        else:
            predefined_input = args.smiles
            if has_selfies:
                predefined_input_selfies = sf.encoder(predefined_input)
                tokens = ["X"] + list(sf.split_selfies(predefined_input_selfies))
                tokens += ["E" for i in range(encoding_seq_length - len(tokens))]
                encoding_input = [[vocabulary.get(token) for token in tokens] for _ in range(batch_size)]
            else:
                smiles_input = [("X" + predefined_input).ljust(encoding_seq_length, "E")]
                encoding_input = [np.array(list(map(vocabulary.get, s))) for s in smiles_input for _ in range(batch_size)]
            v.set_latent_vector_from_inputs(
                    (encoding_input, properties)
            )

        outputs.extend(
                generate_encoded_result(v, encoding_seq_length, start_codon, properties)
        )

    output: np.ndarray
    for i, output in enumerate(outputs):
        all_smiles = convert_to_smiles(output, vocabulary)
        if "E" in all_smiles:
            tokens = all_smiles.split("E")
            smiles = tokens[0]
            if has_selfies:
                smiles = sf.decoder(tokens[0])

            if Chem.MolFromSmiles(smiles) is not None:
                output_smiles.append(smiles)

    print(f"found {len(output_smiles)} valid molecules (with duplicates).")
    output_smiles = list(set(output_smiles))

    print(f"found {len(output_smiles)} valid molecules (without duplicates).")
    if args.output is not None:
        with open(args.output, "w") as smi_file:
            for s in output_smiles:
                smi_file.write(f"{s}\n")
    else:
        for s in output_smiles:
            if args.is_verbose:
                print(s)
