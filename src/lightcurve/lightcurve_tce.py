import pandas as pd

def preprocess_tce(tce_table):

    tce_table = tce_table.drop_duplicates(subset=['tic_id'], keep = "first")
    tce_table = tce_table[tce_table['Transit_Depth'] > 0]
    tce_table["Duration"] /= 24  # Convert hours to days.
    tce_table['Disposition'] = tce_table['Disposition'].replace({'IS': 'J', 'V': 'O'}) #Reduce classification labels [Instrumental Noise -> Junk, Variable Star -> Others]
    #? Disposition rimaste: [Planet Candidate PC (1), Eclipsing Binary EB (0), Junk J (0), Other O (0)]
    tce_table = tce_table.append([tce_table[tce_table['Disposition'] == 'PC']] * 3, ignore_index=True) #? Introduciamo rindondaza che dovrebbe aumentare i casi positivi
    tce_table = tce_table.dropna()
    print(tce_table.info())
    return tce_table


def create_input_list(tce_csv):
    """Generate pandas dataframe of TCEs to be made into file shards.

    :return: pandas dataframe containing TCEs. Required columns: TIC, final disposition
    """
    ready_tce_table = None
    if type(tce_csv) == list:
        tce_table = pd.DataFrame()
        for input in tce_csv: #? Come memorizzare gli ID TIC separatamente per poi aggiungerli dopo la fase di training/test per una migliore comprensione dei dati
            table = pd.read_csv(input, header=0, usecols=[1,3,6,7,8,9,10,11,16],
                                #tic, dispo, tmag, epoc, period, duration, transit, sectors, sn
                                dtype={'Sectors': int})
            tce_table = pd.concat([tce_table, table])
    else:
        tce_table = pd.read_csv(tce_csv, header=0, usecols=[1,3,6,7,8,9,10,11,16],
                                dtype={'Sectors': int})

    ready_tce_table = preprocess_tce(tce_table)

    return ready_tce_table