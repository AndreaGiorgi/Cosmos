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
def training_session(json_config, training_files, validation_files, test_files, model_path):

    model_config = model_util.load_config(json_config) # return config dictionary
    lc_training_dataset, aux_training_dataset = dataset_builder.dataset_builder(training_files, model_config.folder, model_config.inputs, 0.5, 500)
    lc_validation_dataset, aux_validation_dataset = dataset_builder.dataset_builder(validation_files, model_config.folder, model_config.inputs, 0.5, 500)
    lc_test_dataset, aux_test_dataset = dataset_builder.dataset_builder(test_files, model_config.folder, model_config.inputs, 0.5, 500, True)
    cosmos_model =  model_initializer._test_build(lc_training_dataset, aux_training_dataset, lc_validation_dataset, aux_validation_dataset,
                                                        lc_test_dataset, aux_test_dataset, 'test') #return a cosmos model using json hparams

    #cosmos_cnn.train(training_tensors, validation_tensors, max_epochs = 5000)

    return True


if __name__=='__main__':
    training_session('global_model_config.json', 'training_set', 'val', 'test', 'src\\model_checkpoint')