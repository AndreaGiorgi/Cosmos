import os
import sys
import psutil
import time
from neural_network_util import data_ops, model_util
from neural_network import model_initializer
import tensorflow as tf

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


def track(func):
    def wrapper(*args, **kwargs):
        mem_before = get_process_memory()/1024/1024
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start
        mem_after = get_process_memory()/1024/1024
        print("{}: memory before: {:,} MB, after: {:,} MB, consumed: {:,} MB; exec time: {}".format(
            func.__name__,
            mem_before, mem_after, (mem_after - mem_before),
            elapsed_time))
        return result
    return wrapper


@track
def training_session(json_config, training_files, validation_files, model_directory):

    model_config = model_util.load_config(json_config) # return config dictionary
    training_dataset = data_ops.dataset_builder(training_files, model_config.inputs, model_config.shuffle_buffer, reverse = 0.5)

    
    
    
    
    #tensorflow_estimator = tf.estimator.RunConfig(tf_random_seed = 42, keep_checkpoint_max = 1) # return an estimator with max n° of model checkpoints
    #training_tensors = data_util.tensor_function(training_files, config = model_config.inputs, mode = tf.estimator.ModeKeys.TRAIN, shuffle_buffer = model_config.shuffle_buffer,
    #                                          repeat_dataset = 1)
    #validation_tensors = data_util.tensor_function(validation_files, model_config.inputs, tf.estimator.ModeKeys.EVAL, model_config.shuffle_buffer, repeat_dataset = 1)
    #cosmos_cnn = #TODO model_initializer.define_model(model_config.hparams, tensorflow_estimator, model_directory) #return a cosmos model using json hparams

    #cosmos_cnn.train(training_tensors, validation_tensors, max_epochs = 5000)

    return True


@track
def init_evaluation_session(json_config, testing_files, model_directory):

    return True