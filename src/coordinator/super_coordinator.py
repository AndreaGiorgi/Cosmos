# TODO: SPIEGA STO CAZZO DI ACCROCCO

#!/usr/bin/python

#Permette di inserire all'interno del path il programma
import os, sys, psutil, time
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import psutil, argparse, time
import lightcurves_processing_coordinator
from coordinator import etl_coordinator, neural_network_coordinator

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

parser = argparse.ArgumentParser()

parser.add_argument(
    "--tce_csv",
    type=str,
    required=True,
    help="CSV file containing the TESS TCE table. Must contain "
    "columns: row_id, tic_id, toi_id, Period, Duration, "
    "Epoc (t0).")

parser.add_argument(
    "--astro_tce",
    type=str,
    required=False,
    help="CSV file containing the TESS TCE table. Must contain "
    "columns: row_id, tic_id, toi_id, Period, Duration, "
    "Epoc (t0). Required if tce_csv == tev_tces for new training data augmentation"
)

parser.add_argument(
    "--output_directory",
    type=str,
    required=True,
    help="Directory in which to save the output.")

parser.add_argument(
    "--shards",
    type=int,
    default=10,
    help="Number of file shards to divide the training set into.")

parser.add_argument(
    "--only_local",
    action='store_true',
    default=False,
    help="Generate just lightcurve local view rather than global and local ones?")

parser.add_argument(
    "--new_train",
    action='store_true',
    default=False,
    help="Generate additional training set using tev toi-catalog data?")

NAMESPACE, unparsed = parser.parse_known_args()

def multiprocess_params_util():
    avaiable_workers = psutil.cpu_count(logical=True)
    print("Avaiable CPU Cores: ", avaiable_workers)
    dedicated_workers = int(2*avaiable_workers/3)
    print("Dedicated CPU Cores to Preprocessing Phase: ", dedicated_workers)
    return dedicated_workers

@track
def training_pipeline():
    workers = multiprocess_params_util()
    for i in range(1,6):
        sector = str(i)
        etl_coordinator.etl_ingestion.create_sector_folder(sector = sector)
    lightcurves_processing_coordinator.main_train_val_test_set(NAMESPACE.tce_csv, NAMESPACE.output_directory, NAMESPACE.shards, workers, NAMESPACE.only_local)
    # TODO neural_network_coordinator.init_training_session()
    return

@track
def new_data_pipeline():
    workers = multiprocess_params_util()
    if etl_coordinator.start_new_tce_formatting(NAMESPACE.tce_csv, NAMESPACE.astro_tce):
        lightcurves_processing_coordinator.main_new_data_training_set(NAMESPACE.tce_csv, NAMESPACE.output_directory, 
                                                                    NAMESPACE.shards, workers, NAMESPACE.only_local, NAMESPACE.sector)
        #TODO neural_network_coordinator.init_training_session()
    return

def main():
    if NAMESPACE.new_train is False:
        training_pipeline()
    new_data_pipeline()
    
if __name__ == '__main__':
    training_pipeline()