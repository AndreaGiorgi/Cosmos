import pandas as pd
import numpy as np

def load_original_tce(tce_path):
    astro_tce = pd.read_csv(tce_path, sep=',', verbose=False)
    astro_columns = astro_tce.columns
    print(astro_tce.info())
    print(astro_columns)

    return astro_tce, astro_columns

def load_new_tce(tce_path):
    
    tev_tce = pd.read_csv(tce_path, sep = ',', verbose=False)
    print(tev_tce.info())


def format_csv(tev_path, astro_path):
    
    astro_tce, astro_columns = load_original_tce(astro_path)
    tev_tce = load_new_tce(tev_path)
    
    ## column sectors. IF sectors elems >= 2 THEN keep elems[0] AND insert new value sectors = elem[0]

    return True

if __name__ == '__main__':
    format_csv('F:\\Cosmos\\Cosmos\\new_tces.csv', 'F:\\Cosmos\\Cosmos\\tces.csv')