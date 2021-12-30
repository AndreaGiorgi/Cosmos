import os, sys
import glob as g
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import tensorflow as tf
from neural_network_util import data_ops

def create_tfrecords_index(dataset, folder):

    filenames = g.glob("src\\{folder}\\*{dataset}*".format(folder = folder, dataset = dataset))
    index = ','.join(filenames)
    return index


def open_tfrecord():
    raw_dataset = tf.data.TFRecordDataset("src\\TFRecords\\training_set-00001-of-00002")
    for raw_record in raw_dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        print(example)

if __name__ == '__main__':
    #create_tfrecords_index("training_set", "TFRecords")
    open_tfrecord()