
#!/usr/bin/python

#Permette di inserire all'interno del path il programma
import os, sys, psutil, time
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import psutil, time
from coordinator import etl_coordinator, lightcurves_processing_coordinator, coordinator_config

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


def multiprocess_params_util():
    avaiable_workers = psutil.cpu_count(logical=True)
    print("Avaiable CPU Cores: ", avaiable_workers)
    dedicated_workers = int(2*avaiable_workers/3)
    print("Max CPU Cores to data processing phase: ", dedicated_workers)
    return dedicated_workers


@track
def training_data_pipeline(config):
    workers = multiprocess_params_util()
    for i in range(1,5):
        sector = str(i)
        etl_coordinator.etl_ingestion.create_sector_folder(sector = sector)
    lightcurves_processing_coordinator.main_train_val_test_set(config.tce_csv, config.output_directory, config.shards, workers, config.only_local)
    return True


@track
def test_data_pipeline(config):
    workers = multiprocess_params_util()
    etl_coordinator.etl_ingestion.create_sector_folder(sector = 5)
    lightcurves_processing_coordinator.main_test_set(config.tce_csv, config.output_directory, config.shards, workers, config.only_local)
    return True

def main():
    config = coordinator_config.load_config('F:\\Cosmos\\Cosmos\\ETL_config.json')
    if config.etl.test == 1:
        test_data_pipeline(config.etl)
    else:
        training_data_pipeline(config.etl)

if __name__ == '__main__':
    main()