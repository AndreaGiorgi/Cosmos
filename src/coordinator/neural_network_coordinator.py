import os
import sys
import psutil
import time

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from neural_network_util import dataset_builder, model_util
from neural_network import model_initializer

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
def model_build_evaluation(json_config, training_files, validation_files, test_files, local):

    model_config = model_util.load_config(json_config) # return config dictionary
    lc_training_dataset, aux_training_dataset = dataset_builder.dataset_builder(training_files, model_config.folder, model_config.inputs, 0.5, 500)
    lc_validation_dataset, aux_validation_dataset = dataset_builder.dataset_builder(validation_files, model_config.folder, model_config.inputs, 0.5, 500, False)
    lc_test_dataset, aux_test_dataset = dataset_builder.dataset_builder( test_files, model_config.folder, model_config.inputs, 0.5, 500, False)
    cosmos_model =  model_initializer._test_build(local, lc_training_dataset, aux_training_dataset, lc_validation_dataset, aux_validation_dataset,
                                                        lc_test_dataset, aux_test_dataset, model_config.mlp_net, model_config.cnn_net) #return a cosmos model using json hparams



if __name__=='__main__':
    model_build_evaluation('global_model_config.json', 'training_set', 'val', 'test', False)