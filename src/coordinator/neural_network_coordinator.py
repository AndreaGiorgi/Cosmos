# TODO: SPIEGA STO CAZZO DI ACCROCCO

import os, sys, psutil, time
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from neural_network_util import model_util, data_util, data_ops
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
def training_session(json_config, training_files, validation_files, model_directory):
    
    model_config = model_util.load_config(json_config) # return config dictionary
    tensorflow_estimator = tf.estimator.RunConfig(keep_checkpoint_max = 1) # return an estimator with max n° of model checkpoints
    #input_tensors = data_util.input_pipeline(file_names, config = model_config.inputs, mode = tf.estimator.ModeKeys.TRAIN, model_config.shuffle_buffer)
    #cosmos_cnn = model_initializer.define_model(model_config.hparams, run_config, model_directory) #return a cosmos model using json hparams
    
    
    
    return True

@track
def init_evaluation_session(json_config, testing_files, model_directory):
    
    return True