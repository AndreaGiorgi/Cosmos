import os
import sys
import psutil
import time

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from neural_network_util import data_ops, model_util
import tensorflow as tf

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
def training_session(json_config, training_files):

    model_config = model_util.load_config(json_config) # return config dictionary
    training_dataset = data_ops.dataset_builder(training_files, model_config.folder, model_config.inputs, 0.5, 500)

    #training_tensors = neural_network_util.data_ops.dataset_builder(training_files, model_config.inputs, tf.estimator.ModeKeys.TRAIN, model_config.shuffle_buffer) #? returns Dataset
    #validation_tensors = data_ops.dataset_builder(validation_files, model_config.inputs, tf.estimator.ModeKeys.EVAL, model_config.shuffle_buffer)

    #cosmos_cnn = #TODO model_initializer.define_model(model_config.hparams, tensorflow_estimator, model_directory) #return a cosmos model using json hparams

    #cosmos_cnn.train(training_tensors, validation_tensors, max_epochs = 5000)

    return True


@track
def init_evaluation_session(json_config, testing_files, model_directory):

    return True

if __name__=='__main__':
    training_session('model_config.json','training_set')