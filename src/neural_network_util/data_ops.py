import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import collections
import six
import tensorflow as tf
from TFRecords import TFRecords_util


def _recursive_set_batch_size(tensor_or_collection, batch_size):
  """Recursively sets the batch size in a Tensor or collection of Tensors."""
  if isinstance(tensor_or_collection, tf.Tensor):
    t = tensor_or_collection
    shape = t.shape.as_list()
    shape[0] = batch_size
    t.set_shape(t.shape.merge_with(shape))
  elif isinstance(tensor_or_collection, dict):
    for t in six.itervalues(tensor_or_collection):
      _recursive_set_batch_size(t, batch_size)
  elif isinstance(tensor_or_collection, collections.Iterable):
    for t in tensor_or_collection:
      _recursive_set_batch_size(t, batch_size)
  else:
    raise ValueError("Unknown input type: %s" % tensor_or_collection)

  return tensor_or_collection


def set_batch_size(dataset, batch_size):
  """Sets the batch dimension in all Tensors to batch_size."""
  return dataset.map(lambda t: _recursive_set_batch_size(t, batch_size))


def retrive_dataset(tfrecords, config, batch_size, labels, reverse_prob, shuffle_filenames, shuffle_buffer, repeat_dataset):

    tfr_files = TFRecords_util.create_tfrecords_index(tfrecords)

    # Checks that labels ids are integers starting at 0
    label_ids = set(config.label_map.values())
    if label_ids != set(range(len(label_ids))):
        raise ValueError("Label ID not starting from 0")

    # Use an HashTable in order to map label strings to integer ids.
    table_init = tf.lookup.KeyValueTensorInitializer(keys = list(config.label_map.values()), values = list(config.label_map.values()), key_dtype=tf.string, value_dtype=tf.int32)
    label_to_id = tf.lookup.HashTable(table_init, default_value = -1)

    def dataset_parser(tf_example):
        #? Parses a tensorflow example into two tensors [feature, label]

        fields = {
            feature_name: tf.io.FixedLenFeature([feature.lenght], tf.float32)
            for feature_name, feature in config.features.items()
        }

        fields[config.label_feature] = tf.io.FixedLenFeature([], tf.string)

        #? Parse the features.
        parsed_features = tf.io.parse_single_example(tf_example, features=fields)

        #! Data Augmentation 1: Random time series reverse
        reverse = tf.math.less(tf.random.uniform([], 0, 1), reverse_prob, name="should_reverse")

        #? Reorganize
        output = {}
        for feature_name, value in parsed_features.items():
            if labels and feature_name == config.label_feature:
                label_id = label_to_id.lookup(value)

                assert_known_label = tf.debugging.Assert(
                    tf.math.greater_equal(label_id, tf.to_int32(0)),
                    ["Unknown Label:", value])
                with tf.control_dependencies([assert_known_label]):
                    label_id = tf.identity(label_id)

                output["labels"] = label_id
            elif config.features[feature_name].is_time_series:
                        # Possibly reverse.
                if reverse_prob > 0:
                    value = tf.cond(reverse, lambda: tf.reverse(value, axis=[0]), lambda: tf.identity(value))
                if "time_series_features" not in output:
                    output["time_series_features"] = {}
                    output["time_series_features"][feature_name] = value
            else:
                if "aux_features" not in output:
                    output["aux_features"] = {}
                output["aux_features"][feature_name] = value
        return output

    filename_dataset = tf.Data.Dataset.from_tensor_slices(tfr_files) #? Input: List
    if len(tfr_files) > 1 and shuffle_filenames:
        filename_dataset = filename_dataset.shuffle(len(tfr_files))
    dataset = filename_dataset.flat_map(tf.Data.TFRecordsDataset)
    if shuffle_buffer > 0:
        dataset = dataset.shuffle(shuffle_buffer)

    #? Repeat dataset? [#! Possible augmentation]
    if repeat_dataset != 1:
        dataset = dataset.repeat(dataset)

    dataset = dataset.map(dataset_parser, num_parallel_calls=8)
    dataset = dataset.batch(batch_size)

    if repeat_dataset == -1 or repeat_dataset is None:
        # The dataset repeats infinitely before batching, so each batch has the
        # maximum number of elements.
        dataset = set_batch_size(dataset, batch_size)


    # Prefetch a few batches.
    dataset = dataset.prefetch(max(1, int(256 / batch_size)))

    return dataset