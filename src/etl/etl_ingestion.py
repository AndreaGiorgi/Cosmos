import os, time, psutil, shutil
from glob import glob
from astroquery.mast import Observations

class AstroqueryNotWorking(Exception): 
    """Indicates Astroquery not working or not online
	Args:
		Exception Mast connection not established
	"""
    pass

class SectorNotExistingError(Exception):
	"""Indicates sector not existing in sectors' directory
	Args:
		Exception Given sector name is invalid
	"""
	pass

class FitsNotFoundError(Exception):
	"""Indicates fits not existing in sectors' directory
	Args:
		Exception Given fits name is invalid
	"""
	pass

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
def create_sector_folder(sector):
    sector_base_path = 'F:\\Cosmos\\Cosmos\\ingested_data\\sector_' + str(sector)
    num_ingested_files = 0
    if os.path.isdir(sector_base_path):
        return True
    else:
        try:
            os.makedirs(sector_base_path, mode=0o666, exist_ok=True)
            data_path = 'F:\\Cosmos\\Cosmos\\data\\sector_' + str(sector)
            for f in os.scandir(data_path): # dir = sector_i/s000k
                sub_sector = f.path
                for k in os.scandir(sub_sector): # dir = sector_i/s000k/
                    sub_frame = k.path
                    for j in os.scandir(sub_frame): # dir = sector_i/s000k/sub_section_j/
                        sub_section = j.path
                        for z in os.scandir(sub_section): # dir = sector_i/s000k/sub_section_j/sub_sub_section_z
                            sub_sub_section = z.path
                            for x in os.scandir(sub_sub_section): # dir = sector_i/s000k/sub_section_j/sub_sub_section_z/stellar_frame
                                stellar_frame = x.path
                                for filename in glob(os.path.join(stellar_frame, '*.fits')):
                                    shutil.move(filename, sector_base_path)
                                    num_ingested_files += 1
            print("Sector " + str(sector) + " ingested. Lightcurves found and moved into relevant directory: " + str(num_ingested_files))                     
        except OSError as e:
             print("Can't create {dir}: {err}".format(dir=sector_base_path, err=e))
        try:
            os.remove(data_path)
            print("Starting folder removed for memory management. ")
        except OSError as e:
            print("Can't remove {dir}: {err}".format(dir=data_path, err=e)) 
            
@track
def search_lightcurve(tic, sector):
    sector_base_path = 'ingested_data\\'
    try:
        dir = os.path.join(sector_base_path, 'sector_' + str(sector))
        tic_query_key = str(tic)
        try:
            lightcurve_dir = dir + '\\'
            for file in glob(lightcurve_dir + '*'+ tic_query_key +'*'):
                fits_filename = file
        except FitsNotFoundError:
                print(str(tic) + " lightcurve not found.")
    except SectorNotExistingError:
        print(str(sector) + " not found in data directory")
            
    return fits_filename

#TODO
#! IMPORTANTE
def search_lightcurve_online(tic, sector):
    # Usa sector per aprire il settore di riferimento
    # usa tic per avviare una ricerca unificata su tutto la super-directory per trovare il fits
    # ritorna la tabella astropy di riferimento invialo al coordinatore. 
    try:
        obsTable = Observations.query_criteria_async(provenance_name = 'QLP', target_name = int(tic), sequence_number = int(sector))
        try:
            data = Observations.get_product_list_async(obsTable)
            lightcurve = Observations.download_products(data)
        except (FitsNotFoundError, IOError) as e:
             print(e)
    except AstroqueryNotWorking:
        print("Mast or device not online")
        
    return lightcurve
