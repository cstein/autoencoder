import argparse
import csv
import json
import os

import numpy as np
from rdkit import Chem

from vae import VariationalAutoEncoder
import encoding


def generate_encoded_result(autoencoder: VariationalAutoEncoder, encoding_length: int, start_codon, properties) -> np.ndarray:
    vae_results = []
    generated_vector = start_codon[:]

    autoencoder.reset()
    for j in range(encoding_length):
        generated_vector, _ = autoencoder((generated_vector, properties))
        vae_results.append(generated_vector)

    return np.asarray(np.concatenate(vae_results, 1).astype(int).squeeze())


def convert_to_smiles(vector: np.ndarray, char):
    list_char = list(char)
    vector = vector.astype(int)
    return "".join(map(lambda x: list_char[x], vector)).strip()


if __name__ == '__main__':
    # TODO make it so sampling only takes as few parameters as possible
    #      concerning the model (i.e. batch size or RNN options are left
    #      to a config.json file) that is given on the command line and
    #      produced when training happens.
    #      things that are related to sampling (mean, stddev etc) are
    #      of course needed
    ap = argparse.ArgumentParser("Sampler", "python sample.py train.json", "samples from a VAE")
    ap.add_argument("--mean", metavar="NUMBER", default=0.0, type=float,
                    help="mean value to adjust latent space sampling")
    ap.add_argument("--sigma", metavar="NUMBER", default=1.0, type=float,
                    help="standard deviation of latent space sampling")
    ap.add_argument("-i", "--iterations", metavar="NUMBER", type=int, default=10, dest="num_iterations", help="number of iterations to run. Default is %(default)s.")
    ap.add_argument("-f", dest="file", metavar="FILE", help=".json file with configuration.")
    ap.add_argument("-o", "--output", metavar="FILE", default=None, help="The filename (.csv format) to use for output.")
    ap.add_argument("-q", action="store_false", default=True, dest="is_verbose", help="specify to hide output.")
    ap.add_argument("-p", nargs="*", type=float)

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
    backup_folder = settings["backup_folder"]
    backup_checkpoint = settings["backup_checkpoint"]
    backup_numbering = settings["backup_numbering"]
    model_path = os.path.join(backup_folder, backup_checkpoint)
    num_epochs = settings["epochs"]
    full_backup_path = os.path.join(backup_folder, backup_checkpoint)
    if backup_numbering:
        filename, _ = os.path.splitext(backup_checkpoint)
        # we iterate from the last iteration in an attempt to find
        # the file with the _highest_ epoch in the filename
        for i in range(num_epochs, 0, -1):
            potential_model_path = os.path.join(backup_folder, f"{filename}_{i:03d}.ckpt")
            # we check if the .index file of the model is present
            # which signals that this is the model to use.
            if os.path.isfile(potential_model_path + ".index"):
                model_path = potential_model_path
                break

    mean = args.mean
    stddev = args.sigma

    input_smiles = encoding.load_smiles_file(smiles_file)
    can_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(s), True) for s in input_smiles]
    input, output, alphabet, vocabulary, lengths = encoding.encode(can_smiles, encoding_seq_length)

    start_codon = np.array([np.array(list(map(vocabulary.get, 'X'))) for _ in range(batch_size)])
    properties = np.array([[] for _ in range(batch_size)])
    if args.p is not None:
        properties = np.array([args.p for _ in range(batch_size)])

    vocab_size = len(vocabulary)
    # print(input[0])
    outputs = []
    output_smiles = []
    v = VariationalAutoEncoder(latent_size=latent_size, vocab_size=vocab_size, batch_size=batch_size,
                               rnn_num_dimensions=unit_size, rnn_num_layers=rnn_num_layers,
                               mean=mean, stddev=stddev)

    v.load_weights(model_path).expect_partial()

    for i in range(args.num_iterations):
        outputs.extend(generate_encoded_result(v, encoding_seq_length, start_codon, properties))

    output: np.ndarray
    for i, output in enumerate(outputs):
        all_smiles = convert_to_smiles(output, vocabulary)
        if "E" in all_smiles:
            tokens = all_smiles.split("E")

            if Chem.MolFromSmiles(tokens[0]) is not None:
                output_smiles.append(tokens[0])

    print(f"found {len(output_smiles)} valid molecules (with duplicates).")
    output_smiles = set(output_smiles)
    output_smiles = list(output_smiles)
    print(f"found {len(output_smiles)} valid molecules (without duplicates).")
    print(output_smiles)
    if args.output is not None:
        with open(args.output, "w") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["SMILES"])
            for s in output_smiles:
                writer.writerow([s])
    else:
        for s in output_smiles:
            if args.is_verbose:
                print(s)
