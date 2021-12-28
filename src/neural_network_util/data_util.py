import os, sys, psutil, time
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import tensorflow as tf

def dataset_post_processing(dataset, batch_size):
    # Batch
    dataset = dataset.batch(batch_size)
    # Prefetch
    dataset = dataset.prefetch(max(1, int(256 / batch_size)))
    # cazzing
    return dataset