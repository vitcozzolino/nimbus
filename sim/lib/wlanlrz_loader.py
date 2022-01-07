import pandas as pd
import numpy as np
from tqdm import tqdm
from config import hyperparams
from entities.wlanlrz_loader_process import PProcess
from multiprocessing import Queue
import time

def load_data_description(file="full_ap_description.csv"):
    return pd.read_csv(file)

def expand_coordinates(coordinate):
    return coordinate.replace(")", "").replace("(", "").replace("'", "").split(",")

def load_data_parallel(location : list, description, mass_load=False):
    print("Parallel loading of APs data ...")
    if mass_load:
        location = description.Standort.unique()

    filtered_desc = description[(description['Standort'].isin(location))][["ap", "location", "Standort"]]

    filtered_desc['latitude'] = filtered_desc.apply(lambda row: expand_coordinates(row.location)[0], axis = 1) 
    filtered_desc['longitude'] = filtered_desc.apply(lambda row: expand_coordinates(row.location)[1], axis = 1) 
    filtered_desc = filtered_desc.loc[:, ['ap', 'latitude', 'longitude', "Standort"]]    
    filtered_desc = filtered_desc.reset_index(drop=True)

    # Build queues for multithreading
    input_queue = Queue(maxsize=0)
    result_queue = Queue(maxsize=0)

    print("Activating threads now")
    processes = []
    for _,s in filtered_desc.groupby('Standort'):
        input_queue.put(s)
        t = PProcess(input_queue, result_queue)
        processes.append(t)
        t.start()
    
    dfs = []

    # Check if processes are alive (which forces a join() under the hood)
    # Extract data from the queue
    while 1:
        running = any(p.is_alive() for p in processes)
        while not result_queue.empty():
            s = result_queue.get()
            dfs.append(s)
        if not running:
            break
        time.sleep(.1)
    
    # Get standard data for users density
    data = [item[0].total for item in dfs]
    s = np.sum(data)
    data = pd.DataFrame(s)
    data.dropna(inplace=True)
    
    # Get coordinates for heatmap plotting
    coord_dataframe_list = [item[1] for item in dfs]

    # Get the total of APs in the network
    total_ap = np.sum([item[2] for item in dfs])

    # Get raw AP data per timestamp
    raw_ap_data = [item[3] for item in dfs]

    for idx, elem in enumerate(raw_ap_data):
        if idx == 0:
            merged_raw_ap_data = elem
        else:
            merged_raw_ap_data = merged_raw_ap_data.merge(elem, left_index=True, right_index=True)

    return data, coord_dataframe_list, total_ap, merged_raw_ap_data

# DEPRECATED - USE PARALLEL VERSION
def load_data(location : list, description):
    print("Loading data from APs ...")
    filtered_desc = description[(description['Standort'].isin(location))][["ap", "location", "Standort"]]

    filtered_desc['latitude'] = filtered_desc.apply(lambda row: expand_coordinates(row.location)[0], axis = 1) 
    filtered_desc['longitude'] = filtered_desc.apply(lambda row: expand_coordinates(row.location)[1], axis = 1) 
    filtered_desc = filtered_desc.loc[:, ['ap', 'latitude', 'longitude', "Standort"]]    
    filtered_desc = filtered_desc.reset_index(drop=True)

    coord_dataframe_list = []
    prev = None

    for idx, row in tqdm(filtered_desc.iterrows(), total=len(filtered_desc)):
        fd = row.ap
        ap_data = pd.read_csv('{}/{}.gz'.format(hyperparams.FOLDER, fd), compression='gzip', header=0, sep=',', quotechar='"', error_bad_lines=False)

        # Get the total amount of users connected to the AP for all networks
        col_list = list(ap_data)
        col_list.remove('timestamp')
        ap_data.fillna(0, inplace=True)
        ap_data = ap_data.assign(total=np.ceil(ap_data[col_list].sum(axis=1)))

        # Filter the dataframe to output only timestamp and total
        ap_data = ap_data.loc[:, ['timestamp', 'total']]    
        ap_data.drop_duplicates(subset=['timestamp'], inplace=True)
        ap_data.set_index('timestamp', inplace=True)
        #ap_data['timestamp'] = ap_data['timestamp'] // 100
        #ap_data.columns = [str(col) if idx == 0 else fd + "." + str(col) for idx, col in enumerate(ap_data.columns)]
        ap_data.columns = [fd + "." + str(col) for col in ap_data.columns]
        data = ap_data if idx == 0 else pd.concat([data, ap_data], axis=1)
        
        if prev is None or row.Standort == prev:
            coord_data = ap_data if idx == 0 else pd.concat([coord_data, ap_data], axis=1)
        else:
            print("Detected location change")
            coord_data.dropna(inplace=True)
            coord_data = coord_data.assign(total=coord_data.sum(axis=1)).loc[:, ['total']]  
            coord_data = coord_data.assign(latitude=row.latitude)
            coord_data = coord_data.assign(longitude=row.longitude)
            coord_data = coord_data.assign(location=row.Standort)
            coord_dataframe_list.append(coord_data)
            coord_data = pd.DataFrame()
        
        prev = row.Standort
    
    # Cleaning the last dataframe before appending
    coord_data.dropna(inplace=True)
    coord_data = coord_data.assign(total=coord_data.sum(axis=1)).loc[:, ['total']]  
    coord_data = coord_data.assign(latitude=row.latitude)
    coord_data = coord_data.assign(longitude=row.longitude)
    coord_data = coord_data.assign(location=row.Standort)
    coord_dataframe_list.append(coord_data)

    # Sum everything up
    data.dropna(inplace=True)
    data = data.assign(total=data.sum(axis=1)).loc[:, ['total']]  
    total_ap = len(filtered_desc)

    return data, coord_dataframe_list, total_ap