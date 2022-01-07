#%%
import os, time
import numpy as np
import pandas as pd
from tqdm import tqdm
from lib import wlanlrz_loader, rtt_matrix_loader, inference_loader, solver
from entities.solver_process import SolverProcess  
from lib.common import disarrange
import powerlaw
import matplotlib.pyplot as plt
import seaborn as sns
import config.hyperparams as hp
import datetime as dt
from multiprocessing import Queue

### WLAN-LRZ AP Data Loader ####
desc = wlanlrz_loader.load_data_description(hp.AP_DESCR)
data, coord_dataframe_list, total_ap, merged_raw_ap_data = wlanlrz_loader.load_data_parallel(hp.BUILDING, desc, mass_load=False)

print("Total APs: {}".format(total_ap))
data['timestamp'] = data.index

# Filter the dataser based on the minimum amount of users we want to serve
data = data[data.total >= hp.minimum_agents_threshold]
merged_raw_ap_data = merged_raw_ap_data.loc[merged_raw_ap_data.sum(axis=1) >= hp.minimum_agents_threshold]

data = data.sample()
merged_raw_ap_data = merged_raw_ap_data.sample()

TOTAL_EPISODES = int(len(data))
TIER_1_EN = int(round(total_ap/hp.T1_RATIO))
TIER_2_EN = hp.T2_RATI0
TIER_3_EN = hp.T3_RATI0
EN_RATIO = (TIER_1_EN, TIER_2_EN, TIER_3_EN)
TOTAL_EN = int(TIER_1_EN + TIER_2_EN + TIER_3_EN)

if hp.STORE_RESULTS:
    try:
        os.mkdir(hp.CSV_FOLDER)
    except OSError:
        print ("Creation of the directory %s failed" % hp.CSV_FOLDER)
    else:
        print ("Successfully created the directory %s " % hp.CSV_FOLDER)

if hp.dataset_rtt:
    ### LOAD LATENCY MATRIX FROM DATASET ###
    print("Loading rtt matrix from dataset")
    clf = rtt_matrix_loader.analyze_data(source=hp.dataset_rtt, drange=1000, threshold=250, k=3)
    rtt_matrix = rtt_matrix_loader.generate_data(clf, n=TIER_1_EN, m=TOTAL_EN)
else:
    # Prepare latency matrixes for all the EN classes with increasing latency based on distance from the edge 
    rtt_matrix_en_t1 = np.round(abs(np.random.normal(1, 0.2, (TIER_1_EN, TIER_1_EN))))
    rtt_matrix_en_t2 = np.round(abs(np.random.normal(3, 1, (TIER_1_EN, TIER_2_EN))))
    rtt_matrix_en_t3 = np.round(abs(np.random.normal(10, 1, (TIER_1_EN, TIER_3_EN))))

    sns.distplot(rtt_matrix_en_t1.flatten(), hist=False, rug=True)
    sns.distplot(rtt_matrix_en_t2.flatten(), hist=False, rug=True)
    sns.distplot(rtt_matrix_en_t3.flatten(), hist=False, rug=True)

    rtt_matrix = np.hstack([rtt_matrix_en_t1, rtt_matrix_en_t2, rtt_matrix_en_t3])

np.random.shuffle(rtt_matrix)
disarrange(rtt_matrix, axis=0)

print("Plotting and saving RTT matrix")
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
sns.heatmap(rtt_matrix, ax=ax)
fig.tight_layout(w_pad=1.5)
fig.savefig("../plots/rtt_matrix.pdf")

for episode in range(TOTAL_EPISODES):
    # Run the solver
    # M = 10000, N = 300, dim=64, rangee=1, k=10, episode=0
    MA = int(data.total)
    env, agents, edge_nodes, convergence_time = \
        solver.algorithm(MA, EN_RATIO, 5, 4, 100, episode, None, rtt_matrix, merged_raw_ap_data, suppress_output=True)
    
    if hp.STORE_RESULTS:
        env.episodes_tracker[:,:,episode]
        np.savetxt(hp.CSV_FOLDER + '/' + str(episode) + '.csv', env.episodes_tracker[:,:,episode], delimiter=',')

# %%
