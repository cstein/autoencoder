import argparse

from vae import VariationalAutoEncoder

if __name__ == '__main__':
    ap = argparse.ArgumentParser("VAE", "python vae.py my_smiles_file.smi", "Trains the variational autoencoder")

    import tensorflow as tf
    vae = tf.keras.models.load_model("saved/checkpoint.tf")
    print(vae.sample())
    # x = tf.random.normal((10, 200), mean=0.0, stddev=0.0)
    # y = vae(x)
