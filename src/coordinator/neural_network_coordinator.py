# TODO: SPIEGA STO CAZZO DI ACCROCCO

import os, sys, psutil, time
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
def init_training_session(json_config, training_files, validation_files, model_directory):
    
    model_configuration = model_util.load_json(json_config) # return config dictionary
    
    
    
    return True

@track
def init_evaluation_session(json_config, testing_files, model_directory):
    
    return True