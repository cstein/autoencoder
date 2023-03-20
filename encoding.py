import collections

import numpy as np
import pandas as pd
from rdkit import Chem
import selfies as sf


def encode(canonical_smiles: list[str], seq_length: int, ignore_large: bool = False):
    """ Encodes a list of smiles strings with a one-hot style encoding

        :param canonical_smiles: list of unique canonical smiles
        :param seq_length: the length of the sequence encoding
        :param ignore_large: removes large molecules from the population
    """
    all_smiles = "".join(canonical_smiles)
    counter = collections.Counter(all_smiles)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    chars, counts = zip(*count_pairs)
    vocab = dict(zip(chars, range(len(chars))))
    chars += ("E", "X")
    vocab["E"] = len(chars)-2
    vocab["X"] = len(chars)-1
    
    lengths = [len(s) + 1 for s in canonical_smiles]
    smiles_input = [("X"+s).ljust(seq_length, "E") for s in canonical_smiles]
    smiles_output = [s.ljust(seq_length, "E") for s in canonical_smiles]
    encoding_input = np.array([np.array(list(map(vocab.get, s))) for s in smiles_input])
    for i, encoded_state in enumerate(encoding_input, start=1):
        if length_of_encoded_state := len(encoded_state) > seq_length:
            print(f"Entry {i} in input has encoding length {len(encoded_state)} > {seq_length}.")
            if ignore_large:
                pass
            else:
                raise ValueError(f"Encoding length of entry is larger than max allowed.")
    encoding_output = np.array([np.array(list(map(vocab.get, s))) for s in smiles_output])
    return encoding_input, encoding_output, chars, vocab, lengths


def encode_selfies(selfies: list[str], seq_length: int):
    """ Encodes a list of selfies strings with one-hot encoding """
    def build_input_encoding(selfie: str, max_length: int, vocabulary: dict):
        encoded_input = [vocabulary.get("X")] + [vocabulary.get(char) for char in sf.split_selfies(selfie)]
        while len(encoded_input) < max_length:
            encoded_input.append(vocabulary.get("E"))
        return encoded_input

    def build_output_encoding(selfie: str, max_length: int, vocabulary: dict):
        encoded_output = [vocabulary.get(char) for char in sf.split_selfies(selfie)]
        while len(encoded_output) < max_length:
            encoded_output.append(vocabulary.get("E"))
        return encoded_output

    """ Encodes a list of selfies with custom one-hot encoding """
    chars = list(sorted(sf.get_alphabet_from_selfies(selfies)))
    chars += ("E", "X")
    vocab = dict(zip(chars, range(len(chars))))

    lengths = [sf.len_selfies(s) + 1 for s in selfies]
    encoding_inputs = np.array([build_input_encoding(s, seq_length, vocab) for s in selfies])
    encoding_outputs = np.array([build_output_encoding(s, seq_length, vocab) for s in selfies])
    return encoding_inputs, encoding_outputs, chars, vocab, lengths


def load_smiles_file(filename: str) -> list[str]:
    """ Loads a smiles file

        :param filename: the file to read
        :returns: a list of smiles from the file
    """
    smiles = []
    with open(filename, "r") as f:
        for line in f:
            tokens = line.split()
            smiles.append(tokens[0])
    return smiles


def load_csv_file(filename: str) -> pd.DataFrame:
    """ Attempts to load a .csv file. Aborts if a column named SMILES is not found. """
    df = pd.read_csv(filename)
    if "SMILES" not in df:
        raise ValueError(f"A column named 'SMILES' was not found in {filename}")
    return df


if __name__ == '__main__':
    smiles = ["CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1", "C[C@@H]1CC(Nc2cncc(-c3nncn3C)c2)C[C@@H](C)C1",
              "N#Cc1ccc(-c2ccc(O[C@@H](C(=O)N3CCCC3)c3ccccc3)cc2)cc1",
              "CCOC(=O)[C@@H]1CCCN(C(=O)c2nc(-c3ccc(C)cc3)n3c2CCCCC3)C1",
              "N#CC1=C(SCC(=O)Nc2cccc(Cl)c2)N=C([O-])[C@H](C#N)C12CCCCC2"]

    can_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(s), True) for s in smiles]
    input, output, _, _, _ = encode(can_smiles, 100)
    print(input)
