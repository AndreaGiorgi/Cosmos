import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import collections
import six
import glob as g
import random
import tensorflow as tf
from neural_network_util import dataset_postprocess, TFRecords_util
AUTOTUNE = tf.data.AUTOTUNE


def dataset_builder(tfrecords, folder, config, reverse_prob, shuffle_buffer):

    tfr_files = TFRecords_util.create_tfrecords_index(tfrecords, folder)
    file_patterns = tfr_files.split(",")
    filenames = []
    for p in file_patterns:
        matches = g.glob(p)
        if not matches:
            raise ValueError("Found no input files matching %s" % p)
        filenames.extend(matches)
    print("Files: {num}, Pattern: {file_patterns}".format(num = len(filenames), file_patterns=file_patterns))

    # Use an HashTable in order to map label strings to integer ids.
    table_init = tf.lookup.KeyValueTensorInitializer(keys = list(config.label_map.keys()), values = list(config.label_map.values()), key_dtype=tf.string, value_dtype=tf.int32)
    label_to_id = tf.lookup.StaticHashTable(table_init, default_value = -1)

    def dataset_parser(tf_example):
        #? Parses a tensorflow example into a special dict-like structure [feature_name, specs] where feature_name is the key and specs are the values

        fields = {
            feature_name: tf.io.FixedLenFeature([feature.length], tf.float32)
            for feature_name, feature in config.features.items()
        }

        fields[config.label_feature] = tf.io.FixedLenFeature([], tf.string)
        print(fields)
        #? Parse the features.
        parsed_features = tf.io.parse_single_example(tf_example, features=fields)

        #? Reorganize
        output = {}
        for feature_name, value in parsed_features.items():
            print(feature_name)
            if feature_name == config.label_feature:
                label_id = label_to_id.lookup(value)
                output["labels"] = label_id
                print("Label")
                print(output["labels"])
            elif config.features[feature_name].is_time_series == True:
                #! Data Augmentation: Random time series reverse based on probability
                reverse_condition = random.uniform(0, 1)
                if reverse_prob > reverse_condition:
                    value = tf.reverse(value, axis=[0])
                output["time_series_features"] = {}
                output["time_series_features"][feature_name] = value
                print("time_series_features")
                print(output["time_series_features"][feature_name])
            else:
                if "aux_features" not in output:
                    output["aux_features"] = {}
                output["aux_features"][feature_name] = value
                print("aux_features")
                print(output["aux_features"][feature_name])

        return output

    filename_dataset = tf.data.Dataset.from_tensor_slices(filenames) #? Input: List
    if len(filenames) > 1:
        filename_dataset = filename_dataset.shuffle(len(filenames))

    dataset = filename_dataset.flat_map(tf.data.TFRecordDataset)
    if shuffle_buffer > 0:
        dataset = dataset.shuffle(shuffle_buffer, reshuffle_each_iteration= True)

    ignore_order = tf.data.Options()
    ignore_order.exprimental_deterministic = False #? disable order increasing speed
    dataset = dataset.with_options(ignore_order)
    # Map the parser over the dataset.
    dataset = dataset.map(dataset_parser, num_parallel_calls=AUTOTUNE)
    dataset = data_util.dataset_post_processing(dataset, AUTOTUNE, 64)

    return dataset