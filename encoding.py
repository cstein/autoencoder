import collections

import numpy as np

import rdkit
from numpy import void
from rdkit import Chem


def encode(can_smiles: list[str], seq_length, ignore_large=False):
    """ Encodes a list of smiles strings """
    all_smiles = "".join(can_smiles)
    counter = collections.Counter(all_smiles)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    chars, counts = zip(*count_pairs)
    vocab = dict(zip(chars, range(len(chars))))
    chars += ("E", "X")
    vocab["E"] = len(chars)-2
    vocab["X"] = len(chars)-1
    
    lengths = [len(s)+1 for s in can_smiles]
    smiles_input = [("X"+s).ljust(seq_length, "E") for s in can_smiles]
    smiles_output = [s.ljust(seq_length, "E") for s in can_smiles]
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


def load_smiles_file(filename: str) -> list[str]:
    smiles = []
    with open(filename, "r") as f:
        for line in f:
            tokens = line.split()
            smiles.append(tokens[0])
    return smiles



if __name__ == '__main__':
    smiles = ["CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1", "C[C@@H]1CC(Nc2cncc(-c3nncn3C)c2)C[C@@H](C)C1",
              "N#Cc1ccc(-c2ccc(O[C@@H](C(=O)N3CCCC3)c3ccccc3)cc2)cc1",
              "CCOC(=O)[C@@H]1CCCN(C(=O)c2nc(-c3ccc(C)cc3)n3c2CCCCC3)C1",
              "N#CC1=C(SCC(=O)Nc2cccc(Cl)c2)N=C([O-])[C@H](C#N)C12CCCCC2"]

    can_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(s), True) for s in smiles]
    input, output, _, _, _ = encode(can_smiles, 100)
    print(input)
