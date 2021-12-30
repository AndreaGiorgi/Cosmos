import tensorflow as tf
from tensorflow import keras

def callbacks_builder():
    initial_learning_rate = 0.01
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=20, decay_rate=0.96, staircase=True)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("tess_prova.h5", save_best_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True)

    return lr_schedule, checkpoint_cb, early_stopping_cb

def model_builder(hparmas):

    lr, checkpoint, early_stopping = callbacks_builder()

    return

