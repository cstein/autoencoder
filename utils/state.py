import tensorflow as tf


def save(model: tf.keras.Model, folder: str):
    check_point = tf.train.Checkpoint(model)
    return check_point.save(folder)


def restore(model: tf.keras.Model, folder: str):
    check_point = tf.train.Checkpoint(model)
    check_point.restore(folder)