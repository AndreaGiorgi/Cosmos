from astroquery.query import to_cache
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
    return tev_tce


def format_csv(tev_path, astro_path):
    
    astro_tce, astro_columns = load_original_tce(astro_path)
    tev_tce = load_new_tce(tev_path)
    print(astro_tce.head())
    ## column sectors. IF sectors elems >= 2 THEN keep elems[0] AND insert new value sectors = elem[0]
    cols = [0, 4, 8, 9, 10, 12, 14, 16, 18, 20, 21, 22, 23, 25, 26, 27, 28, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]
    tev_tce = tev_tce.drop(tev_tce.columns[cols], axis=1)
    print(tev_tce.info())
    tev_tce['camera'] = np.random.randint(1,4, 4703) #Random camera value
    tev_tce['ccd'] = np.random.randint(1,4, 4703) #Random ccd value
    tev_tce['star_mass'] = np.random.uniform(0.5853, 8.23, 4703) #Random Uniform star_mass value
    tev_tce['Qingress'] = np.random.uniform(-1.43052, 1, 4703) #Random uniform Qingress value
    tev_tce['logg'] = np.random.uniform(1.001, 5.001, 4703) #Random uniform Log g value
    tev_tce = tev_tce.rename(columns={"TIC": "tic_id", "Full TOI ID": "toi_id", "TOI Disposition": "Disposition",
                            "TIC Right Ascension": "RA", "TIC Declination": "Dec", "TMag Value": "Tmag",
                            "Epoch Value": "Epoc", "Orbital Period Value": "Period", "Orbital Duration Value": "Duration",
                            "Transit Duration Value": "Duration", "Transit Depth Value": "Transit_Depth", 
                            "Star Radius Value": "star_rad", "Effective Temperature Value": "teff",
                            "Signal-to-noise": "SN"})
    tev_tce['Sectors'] = tev_tce['Sectors'].str.replace('[','')
    tev_tce['Sectors'] = tev_tce['Sectors'].str.replace(']','')
    tev_tce['Sectors'] = tev_tce['Sectors'].str.split().str[0]
    tev_tce['Sectors'] = tev_tce['Sectors'].str.replace(',','')
    print(tev_tce.info())
    print(tev_tce.head())
    tev_tce.to_csv("F:\\Cosmos\\Cosmos\\updated_tces.csv", index=False)
    tev_tce_path = "F:\\Cosmos\\Cosmos\\updated_tces.csv"
    return tev_tce_path

if __name__ == '__main__':
    format_csv('F:\\Cosmos\\Cosmos\\new_tces.csv', 'F:\\Cosmos\\Cosmos\\tces.csv')