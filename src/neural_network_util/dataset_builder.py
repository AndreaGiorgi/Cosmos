import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import glob as g
import random
import tensorflow as tf
import numpy as np
from neural_network_util import TFRecords_util, dataset_postprocess
AUTOTUNE = tf.data.AUTOTUNE


def dataset_builder(tfrecords, folder, config, reverse_prob, shuffle_buffer, test = False):
    lc_dataset = lc_dataset_builder(tfrecords, folder, config, reverse_prob, shuffle_buffer, test)
    aux_datset = aux_dataset_builder(tfrecords, folder, config, shuffle_buffer, test)
    return lc_dataset, aux_datset


def aux_dataset_builder(tfrecords, folder, config, shuffle_buffer, test):
        tfr_files = TFRecords_util.create_tfrecords_index(tfrecords, folder)
        file_patterns = tfr_files.split(",")
        filenames = []
        for p in file_patterns:
            matches = g.glob(p)
            if not matches:
                raise ValueError("Found no input files matching %s" % p)
            filenames.extend(matches)

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
            #? Parse the features.
            parsed_features = tf.io.parse_single_example(tf_example, fields)

            #? Reorganize
            output = {}
            for feature_name, value in parsed_features.items():
                if feature_name == config.label_feature:
                   label_id = label_to_id.lookup(value)
                   label_id = tf.reshape(label_id, [1])
                   output["targets"] = label_id
                elif config.features[feature_name].is_time_series == 1:
                    continue
                else:
                    if "inputs" not in output:
                        output["inputs"] = {}
                    output["inputs"][feature_name] = value
            return output

        filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)
        dataset = filename_dataset.flat_map(tf.data.TFRecordDataset)
        #dataset =  tf.data.TFRecordDataset(filename_dataset)#? Input: List
        if shuffle_buffer > 0:
            dataset = dataset.shuffle(shuffle_buffer, reshuffle_each_iteration= True)

        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False #? disable order increasing speed
        dataset = dataset.with_options(ignore_order)
        # Map the parser over the dataset.
        dataset = dataset.map(dataset_parser, num_parallel_calls=AUTOTUNE)
        dataset = dataset_postprocess.post_build_ops(dataset, AUTOTUNE, 64, test)

        return dataset

def lc_dataset_builder(tfrecords, folder, config, reverse_prob, shuffle_buffer, test):

    tfr_files = TFRecords_util.create_tfrecords_index(tfrecords, folder)
    file_patterns = tfr_files.split(",")
    filenames = []
    for p in file_patterns:
        matches = g.glob(p)
        if not matches:
            raise ValueError("Found no input files matching %s" % p)
        filenames.extend(matches)

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
        #? Parse the features.
        parsed_features = tf.io.parse_single_example(tf_example, fields)

        #? Reorganize
        output = {}
        for feature_name, value in parsed_features.items():
            if feature_name == config.label_feature:
                label_id = label_to_id.lookup(value)
                label_id = tf.reshape(label_id, [1])
                output["targets"] = label_id
            elif config.features[feature_name].is_time_series == 1:
                #! Data Augmentation: Random time series reverse based on probability
                reverse_condition = random.uniform(0, 1)
                if reverse_prob > reverse_condition:
                    value = tf.reverse(value, axis=[0])
                output["inputs"] = {}
                output["inputs"][feature_name] = value

        return output

    filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)
    #dataset = filename_dataset.flat_map(tf.data.TFRecordDataset)
    dataset =  tf.data.TFRecordDataset(filename_dataset)#? Input: List
    if shuffle_buffer > 0:
        dataset = dataset.shuffle(shuffle_buffer, reshuffle_each_iteration= True)

    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False #? disable order increasing speed
    dataset = dataset.with_options(ignore_order)
    # Map the parser over the dataset.
    dataset = dataset.map(dataset_parser, num_parallel_calls=AUTOTUNE)
    dataset = dataset_postprocess.post_build_ops(dataset, AUTOTUNE, 64, test)

    return dataset