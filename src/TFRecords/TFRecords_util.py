import os, sys, psutil, time
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import tensorflow as tf
from neural_network_util import data_ops

def create_tfrecords_index(dataset):
    index = None
    #TODO
    return index


def open_tfrecord():
    raw_dataset = tf.data.TFRecordDataset("src\\TFRecords\\test-00000-of-00001")
    for raw_record in raw_dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        print(example)

if __name__ == '__main__':
    open_tfrecord()