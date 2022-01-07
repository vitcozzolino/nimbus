import threading
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from lib import solver
from entities.env_tracker import EnvTracker
import config.hyperparams as hp
from entities.mobile_agent import Offloaded
from multiprocessing import Process, Queue

class PProcess(Process):
    def __init__(self, input_queue, result_queue):
        super(PProcess, self).__init__()
        self.input_queue =input_queue
        self.result_queue = result_queue
        self.filtered_desc = self.input_queue.get()
        self.data = None
        self.coord_data = None

    def run(self):
        missing_ap = 0
        coord_dataframe_list = []
        prev = None

        for idx, row in tqdm(self.filtered_desc.iterrows(), total=len(self.filtered_desc)):
            try:
                fd = row.ap
                ap_data = pd.read_csv('{}/{}.gz'.format(hp.FOLDER, fd), compression='gzip', header=0, sep=',', quotechar='"', error_bad_lines=False)

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
                self.data = ap_data if idx == 0 else pd.concat([self.data, ap_data], axis=1)
                
                if prev is None or row.Standort == prev:
                    self.coord_data = ap_data if idx == 0 else pd.concat([self.coord_data, ap_data], axis=1)
                else:
                    print("Detected location change")
                    self.coord_data.dropna(inplace=True)
                    self.coord_data = self.coord_data.assign(total=self.coord_data.sum(axis=1)).loc[:, ['total']]  
                    self.coord_data = self.coord_data.assign(latitude=row.latitude)
                    self.coord_data = self.coord_data.assign(longitude=row.longitude)
                    self.coord_data = self.coord_data.assign(location=row.Standort)
                    coord_dataframe_list.append(self.coord_data)
                    self.coord_data = pd.DataFrame()
                
                prev = row.Standort
            except FileNotFoundError:
                print("File not found!")
                missing_ap += 1
                pass
        
        NoneType = type(None)
        if not isinstance(self.data, NoneType):
            # Cleaning the last dataframe before appending
            self.coord_data.dropna(inplace=True)
            self.coord_data = self.coord_data.assign(total=self.coord_data.sum(axis=1)).loc[:, ['total']]  
            self.coord_data = self.coord_data.assign(latitude=row.latitude)
            self.coord_data = self.coord_data.assign(longitude=row.longitude)
            self.coord_data = self.coord_data.assign(location=row.Standort)
            coord_dataframe_list.append(self.coord_data)

            # Sum everything up (sum-up data from each AP for each timestamp)
            self.data.dropna(inplace=True)
            raw_ap_data = self.data

            self.data = self.data.assign(total=self.data.sum(axis=1)).loc[:, ['total']]  
            total_ap = len(self.filtered_desc) - missing_ap

            self.result_queue.put([self.data, coord_dataframe_list, total_ap, raw_ap_data])