import argparse

import numpy as np
from rdkit import Chem

from vae import VariationalAutoEncoder
import encoding


def generate_encoded_result(encoder: VariationalAutoEncoder, encoding_length: int, start_codon) -> np.ndarray:
    vae_results = []
    generated_vector = start_codon[:]

    encoder.reset()
    for j in range(encoding_length):
        generated_vector, _ = v(generated_vector)
        vae_results.append(generated_vector)

    return np.asarray(np.concatenate(vae_results, 1).astype(int).squeeze())


def convert_to_smiles(vector: np.ndarray, char):
    list_char = list(char)
    vector = vector.astype(int)
    return "".join(map(lambda x: list_char[x], vector)).strip()


if __name__ == '__main__':
    ap = argparse.ArgumentParser("VAE", "python vae.py my_smiles_file.smi", "Trains the variational autoencoder")
    ap.add_argument("--mean", metavar="NUMBER", default=0.0, type=float,
                    help="mean value to adjust latent space sampling")
    ap.add_argument("--sigma", metavar="NUMBER", default=1.0, type=float,
                    help="standard deviation of latent space sampling")
    ap.add_argument("-i", "--iterations", metavar="NUMBER", type=int, default=10, dest="num_iterations", help="number of iterations to run. Default is %(default)s.")
    ap.add_argument("file", metavar="FILE", help="file with SMILES")
    decoder_ap = ap.add_argument_group("Decoder")
    decoder_ap.add_argument("--batch-size", metavar="SIZE", default=1, type=int,
                    help="Size of the batch when training. Default is %(default)s.")
    rnn_ap = ap.add_argument_group("RNN Layers", "Controls options regarding how the RNN is used.")
    rnn_ap.add_argument("--rnn-layers", metavar="NUMBER", dest="rnn_layers_size", default=3,
                        help="Number of layers in the RNN. Default is %(default)s.")
    rnn_ap.add_argument("--rnn-unit-size", metavar="NUMBER", dest="rnn_unit_size", default=512,
                        help="size of the RNN. Default is %(default)s.")
    args = ap.parse_args()
    print(args)
    latent_size = 200
    batch_size = args.batch_size
    # vocab_size = 35
    unit_size = args.rnn_unit_size
    encoding_seq_length = 110
    mean = args.mean
    stddev = args.sigma

    input_smiles = encoding.load_smiles_file(args.file)
    can_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(s), True) for s in input_smiles]
    input, output, alphabet, vocabulary, lengths = encoding.encode(can_smiles, encoding_seq_length)

    start_codon = np.array([np.array(list(map(vocabulary.get, 'X'))) for _ in range(batch_size)])
    vocab_size = len(vocabulary)
    # print(input[0])
    outputs = []
    output_smiles = []
    v = VariationalAutoEncoder(latent_size=latent_size, vocab_size=vocab_size, batch_size=batch_size,
                               unit_size=unit_size, mean=mean, stddev=stddev)

    v.load_weights("saved/checkpoint").expect_partial()

    for i in range(args.num_iterations):
        outputs.extend(generate_encoded_result(v, encoding_seq_length, start_codon))

    output: np.ndarray
    for i, output in enumerate(outputs):
        all_smiles = convert_to_smiles(output, vocabulary)
        if "E" in all_smiles:
            tokens = all_smiles.split("E")

            # if tokens[0] in output_smiles:
            #     continue

            if Chem.MolFromSmiles(tokens[0]) is not None:
                output_smiles.append(tokens[0])

    print(f"found {len(output_smiles)} valid molecules (with duplicates).")
    output_smiles = set(output_smiles)
    output_smiles = list(output_smiles)
    print(f"found {len(output_smiles)} valid molecules (without duplicates).")
    print(output_smiles)
