## Coordinatore delle operazioni di trasformazione, permette di effettuare il processamento in batch dei dati. 
# TODO: SPIEGA STO CAZZO DI ACCROCCO
#! FUNZIONA? BOH SPERAMO 

import os, sys, psutil, time
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import multiprocessing
import numpy as np
import pandas as pd
import tensorflow as tf
import psutil, time
import etl_coordinator

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

class EmptyLightCurveError(Exception):
    """Indicates light curve with no points in chosen time range."""

class SparseLightCurveError(Exception):
    """Indicates light curve with too few points in chosen time range."""

def _process_tce(tce, only_local):
    """Processes the light curve for a Kepler TCE and returns an Example proto.

  Args:
    tce: Row of the input TCE table.
    only_local: Boolean for switching to AstroNet Implement and Cosmos' one

  Returns:
    A tensorflow.train.Example proto containing TCE features.

  Raises:
    IOError or general Exceptionx: If the light curve files for this Kepler ID cannot be found.
  """
    try:
        processed_tce = None
        processed_tce = etl_coordinator.start_processing_phase(tce, only_local)
    except (Exception, IOError) as e:
        print("Exception occurred: ", e)
    return processed_tce

def preprocess_tce(tce_table):
    
    tce_table = tce_table.dropna()
    tce_table = tce_table.drop(['row_id'],  axis=1)
    tce_table = tce_table.drop_duplicates(subset=['tic_id'])
    
    tce_table = tce_table[tce_table['Transit_Depth'] > 0]
    tce_table["Duration"] /= 24  # Convert hours to days.
    tce_table['Disposition'] = tce_table['Disposition'].replace({'IS': 'J', 'V': 'J'}) #Reduce classification labels
    
    return tce_table 

def create_input_list(tce_csv):
    """Generate pandas dataframe of TCEs to be made into file shards.

    :return: pandas dataframe containing TCEs. Required columns: TIC, final disposition
    """
    ready_tce_table = None
    if type(tce_csv) == list:
        tce_table = pd.DataFrame()
        for input in tce_csv:
            table = pd.read_csv(input, header=0, usecols=[0,2,5,6,7,8,9,10,15], 
                                #tic, dispo, tmag, epoc, period, duration, transit, sectors, sn
                                dtype={'Sectors': int})
            tce_table = pd.concat([tce_table, table])
    else:
        tce_table = pd.read_csv(tce_csv, header=0, usecols=[0,2,5,6,7,8,9,10,15],
                                dtype={'Sectors': int})
            
    ready_tce_table = preprocess_tce(tce_table)                                   
    
    return ready_tce_table

#* Processes a single file shard, writing the processed ones into the output file_name
#* Args:
#*  tce_table: A Pandas DataFrame containing the TCEs in the shard
#*  file_name: The output TFRecord file. 
#
def _process_file_shard(tce_table, file_name, only_local = False):
    """Processes a single file shard.

  Args:
    tce_table: A Pandas DateFrame containing the TCEs in the shard.
    file_name: The output TFRecord file.
  """
    with tf.io.TFRecordWriter(file_name) as writer:
        for _, tce in tce_table.iterrows():
            try:
                tce_to_write = _process_tce(tce, only_local)
            except(IOError, EmptyLightCurveError, SparseLightCurveError):
                continue
            if tce_to_write is not None:
                writer.write(tce_to_write.SerializeToString())
                

@track
def main_test_set(tce_csv, output_directory, shards, workers, only_local):
    """
    Make a table of test TCEs into tensorflow.train.Example format for classification purposes. This is useful for
    creating test sets of TCEs from new sectors that were not used to train the model.
    :return:
    """
    # Make the output directory if it doesn't already exist.
    tf.io.gfile.makedirs(output_directory)
    tce_table = create_input_list(tce_csv) 
    num_tces = len(tce_csv)

    # Randomly shuffle the TCE table.
    np.random.seed(2110)
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
        pool.apply_async(_process_file_shard, args = (file_shard), kwds={'only_local': only_local})
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
    tce_table = create_input_list(tce_csv) 
    num_transits = len(tce_table)
    
    # Shuffle TCE Table
    np.random.seed(42429)
    tce_table = tce_table.iloc[np.random.permutation(num_transits)] # SLower than sklearn.shuffle but it now doesn't need to reset the index
    
    ##* TCE Partions:
    #* Training set: 80% of TCEs
    #* Validation set: 10% of TCEs
    #* Test set: 10% of TCEs
    
    train_portion = int(0.90 * num_transits)
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
    #for file_shard in list_of_shards:
    #    _process_file_shard(file_shard)
    
    # Use multiprocessing. One subprocess for each shard
    num_processes = min(num_shards, workers)
    pool = multiprocessing.Pool(processes = num_processes)
    async_results = [
        pool.apply_async(_process_file_shard, args = (file_shard), kwds={'only_local': only_local})
        for file_shard in list_of_shards] 
    pool.close()
    
    for result in async_results:
        result.get()
    
    #.\super_coordinator.py --tce_csv C:\Users\andre\Desktop\tces.csv --output_directory F:\Cosmos\Cosmos\TFRecords