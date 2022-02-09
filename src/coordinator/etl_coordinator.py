import os, sys, psutil, time, pandas as pd
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from etl import etl_ingestion, etl_loading, etl_processing

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
def start_processing_phase(tce, only_local):
    print("Processing Phase: From TCE data to tf.Example proto\n")
    tensorflow_example_proto = etl_processing.process_lightcurve(tce, only_local)
    return tensorflow_example_proto

@track
def start_ingestion_phase(tic, sector):
    print("Loading {tic} fits file from Sector: {sector} \n".format(tic=tic, sector=sector))
    fits_file = etl_ingestion.search_lightcurve(tic, sector)
    return fits_file


@track
def start_loading_phase(fits):
    print("Loading fits data\n")
    lc_time, lc_flux = etl_loading.load_lightcurve_data(fits)
    return lc_time, lc_flux


if __name__ == '__main__':
    start_ingestion_phase(114990015, 1)