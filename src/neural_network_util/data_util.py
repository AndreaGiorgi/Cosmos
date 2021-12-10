import tensorflow as tf
import copy
from data_ops import retrive_dataset

def tensor_function(tfrecords, config, mode, shuffle_buffer, repeat_dataset):
    
    labels = (mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL])
    reverse_time_series_prob = 0.5 if mode == tf.estimator.ModeKeys.TRAIN else 0
    shuffle_filenames = (mode == tf.estimator.ModeKeys.TRAIN)
    
    
    def tensor_fn(config, params):      
        dataset = data_ops.retrive_dataset(tfrecords, config, params["batch_size"], include_labels, reverse_time_series_prob, shuffle_filenames, shuffle_buffer, repeat_dataset)
        return dataset
    
    return tensor_fn

