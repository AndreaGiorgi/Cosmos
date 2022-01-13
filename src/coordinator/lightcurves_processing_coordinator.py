## Coordinatore delle operazioni di trasformazione, permette di effettuare il processamento in batch dei dati.
import os, sys, psutil, time
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import multiprocessing
import numpy as np
import pandas as pd
import tensorflow as tf
import psutil, time
from lightcurve import lightcurve_multiprocessing, lightcurve_tce


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
def main_test_set(tce_csv, output_directory, shards, workers, only_local):
    """
    Make a table of test TCEs into tensorflow.train.Example format for classification purposes. This is useful for
    creating test sets of TCEs from new sectors that were not used to train the model.
    :return:
    """
    # Make the output directory if it doesn't already exist.
    tf.io.gfile.makedirs(output_directory)
    tce_table = lightcurve_tce.create_input_list(tce_csv)
    num_tces = len(tce_csv)

    # Randomly shuffle the TCE table.
    np.random.seed(1803)
    tce_table = tce_table.iloc[np.random.permutation(num_tces)]

    # Further split training TCEs into file shards.
    list_of_shards = []  # List of (tce_table_shard, file_name).
    boundaries = np.linspace(0, len(tce_table), shards + 1).astype(np.int)
    for i in range(shards):
      start = boundaries[i]
      end = boundaries[i + 1]
      list_of_shards.append((tce_table[start:end], os.path.join(
          output_directory, "test-%.5d-of-%.5d" % (i, shards))))

    # Use multiprocessing. One subprocess for each shard
    num_processes = min(len(list_of_shards), workers)
    pool = multiprocessing.Pool(processes = num_processes)
    async_results = [
        pool.apply_async(lightcurve_multiprocessing.process_file_shard, args = (file_shard), kwds={'only_local': only_local})
        for file_shard in list_of_shards]
    pool.close()

    for result in async_results:
        result.get()


@track
def main_train_val_test_set(tce_csv, output_directory, shards, workers, only_local):
    """
    Make a table of new TCEs into tensorflow.train.Example format for classification purposes. This is useful for
    creating test sets of TCEs from new sectors that were not used to train the model.
    :return:
    """
    # Make the output directory if it doesn't already exist.
    tf.io.gfile.makedirs(output_directory)
    tce_table = lightcurve_tce.create_input_list(tce_csv)
    num_transits = len(tce_table)

    # Shuffle TCE Table
    np.random.seed(42429)
    tce_table = tce_table.iloc[np.random.permutation(num_transits)] # SLower than sklearn.shuffle but it now doesn't need to reset the index

    ##* TCE Partions:
    #* Training set: 95% of TCEs
    #* Validation set: 5% of TCEs

    train_portion = int(0.95 * num_transits)
    training_TCEs = tce_table[0:train_portion]
    validation_TCEs = tce_table[train_portion:]

    print(
      "Partitioned {num} TCEs into training ({train}) and validation ({val}))"
      .format(num = num_transits, train = len(training_TCEs), val = len(validation_TCEs)))

    ##* Sharding of Datasets

    list_of_shards = [] #! (tce_table_shard, file_name)
    limits = np.linspace(0, len(training_TCEs), shards + 1).astype(np.int) #Return evenly spaced numbers over a specified interval.

    # Create a list of shards appending the training TCE and its number in list of shards. This allows to create right away a batching set.
    for shard in range(shards):
        start = limits[shard]
        end = limits[shard + 1]
        list_of_shards.append((training_TCEs[start:end], os.path.join(output_directory, "training_set-%.5d-of-%.5d" % (shard, shards))))


    # Test and Validation sets since they represent 20% of all TCEs they are split in only two shards
    list_of_shards.append((validation_TCEs, os.path.join(output_directory,
                                             "val-00000-of-00001")))

    num_shards = len(list_of_shards)
    print(list_of_shards)

    # Use multiprocessing. One subprocess for each shard
    num_processes = min(num_shards, workers)
    pool = multiprocessing.Pool(processes = num_processes)
    async_results = [
        pool.apply_async(lightcurve_multiprocessing.process_file_shard, args = (file_shard), kwds={'only_local': only_local})
        for file_shard in list_of_shards]
    pool.close()

    for result in async_results:
        result.get()

    #.\super_coordinator.py --tce_csv C:\Users\andre\Desktop\tces.csv --output_directory F:\Cosmos\Cosmos\TFRecords