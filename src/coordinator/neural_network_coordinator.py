import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import psutil
import time
from neural_network_util import config_util, dataset_builder, dataset_postprocess
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
def hybrid_dataset_formatter_init(type, dataset):
    return dataset_postprocess.hybrid_dataset_formatter(type, dataset)

@track
def hybrid_dataset_augmentation_init(dataset):
    return dataset_postprocess.hybrid_dataset_augmentation(dataset)

@track
def start_dataset_postprocessing(model_type, dataset, train = False):

    if model_type == 'cnn':
        x, y = dataset_postprocess._lc_dataset_formatter(dataset, train)
    else:
        x, y = dataset_postprocess._aux_dataset_formatter(dataset, train)

    return x, y


@track
def model_build_evaluation(view_config = 'global_model_config.json', cnn_config = 'cnn_config.json', snn_config = 'snn_config.json', hybrid_config = 'hybrid_config.json', training_files = 'training_set', validation_files = 'val', test_files = 'test'):

    view_config = config_util.load_config(view_config) # return config dictionary
    snn_config = config_util.load_config(snn_config) # return config dictionary
    cnn_config = config_util.load_config(cnn_config) # return config dictionary
    hybrid_config = config_util.load_config(hybrid_config)
    lc_training_dataset, aux_training_dataset = dataset_builder.dataset_builder(training_files, view_config.folder, view_config.inputs, 0.5, 500)
    lc_validation_dataset, aux_validation_dataset = dataset_builder.dataset_builder(validation_files, view_config.folder, view_config.inputs, 0.5, 500, False)
    lc_test_dataset, aux_test_dataset = dataset_builder.dataset_builder(test_files, view_config.folder, view_config.inputs, 0.5, 500, False)
    analytics = model_initializer._test_build(lc_training_dataset, aux_training_dataset, lc_validation_dataset, aux_validation_dataset,
                                                        lc_test_dataset, aux_test_dataset, snn_config, cnn_config, hybrid_config) #return a cosmos model using json hparams

    return True


if __name__=='__main__':
    model_build_evaluation('global_model_config.json', 'cnn_config.json', 'snn_config.json', 'hybrid_config.json', 'training_set', 'val', 'test')