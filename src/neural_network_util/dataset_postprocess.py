import os, sys, psutil, time
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import tensorflow as tf

def post_build_ops(dataset, prefetch, batch_size):
    dataset = dataset.shuffle(2110)
    # Batch
    dataset = dataset.batch(batch_size)
    # Prefetch
    dataset = dataset.prefetch(buffer_size = prefetch)
    return dataset