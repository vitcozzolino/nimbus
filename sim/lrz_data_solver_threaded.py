#%%
import os, time
import numpy as np
import pandas as pd
from tqdm import tqdm
from lib import wlanlrz_loader, rtt_matrix_loader, inference_loader
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
# merged_raw_ap_data = merged_raw_ap_data.loc[merged_raw_ap_data.sum(axis=1) >= hp.minimum_agents_threshold]
merged_raw_ap_data = merged_raw_ap_data[merged_raw_ap_data.index.isin(data.index)]

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
    rtt_matrix = rtt_matrix_loader.generate_data(clf, n=total_ap, m=TOTAL_EN)
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

# Manually setting latency for colocated AP/EN
for i in range(round(TIER_1_EN/hp.T1_RATIO)):
    rtt_matrix[i][i] = 1
            
data["mean_task_latency"] = np.nan
data["mean_battery_usage"] = np.nan
data["convergence_time"] = np.nan

# Build queues for multithreading
input_queue = Queue(maxsize=0)
result_queue = Queue(maxsize=0)

# Split original dataframes in n chunks
processes = []
data_t = np.array_split(data, hp.parallel_solvers)
merged_raw_ap_data_t = np.array_split(merged_raw_ap_data, hp.parallel_solvers)

print("Starting parallel solvers")
for d, mrad in zip(data_t, merged_raw_ap_data_t):
    input_queue.put(d)
    t = SolverProcess(
        input_queue, result_queue, mrad,
        len(d), TOTAL_EPISODES, EN_RATIO, rtt_matrix,
        TIER_1_EN, TIER_2_EN, TIER_3_EN
        )
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

data = pd.concat(dfs)

# %%
data.sort_index(inplace=True)
data["datetime"] = pd.to_datetime(data['timestamp'], unit='s')

############# PLOTS: GROUP 1 #############

fig, ax = plt.subplots(6, 1, figsize=(16, 16))

### Plot showing the mean task execution time for each agent
sns.lineplot(x="datetime", y="mean_task_latency", data=data, ax=ax[0], dashes=[6, 2])
sns.lineplot(x="datetime", y=np.median(data["mean_task_latency"]), c='#CC4F1B',data=data, ax=ax[0], dashes=True)
ax[0].fill_between(
    data["datetime"],
    np.median(data["mean_task_latency"])-np.std(data["mean_task_latency"]),
    np.median(data["mean_task_latency"])+np.std(data["mean_task_latency"]),
    alpha=0.2,
    color="#CC4F1B",
    linestyle='dashdot', antialiased=True
    )
ax[0].set(title="Mean task latency", ylabel='ms', xlabel='Time')
ax2 = ax[0].twinx()

color = 'tab:red'
sns.lineplot(x="datetime", y="total_agents", c=color,data=data, ax=ax2, dashes=True)
ax2.set_ylabel('MA', color=color)
ax2.tick_params(axis='y', labelcolor=color)

### Plot showing the mean battery saving for each agent compared to running locally
sns.lineplot(x="datetime", y="mean_battery_usage", data=data, ax=ax[1])
sns.lineplot(x="datetime", y=np.median(data["mean_battery_usage"]), data=data, ax=ax[1], dashes=True)
ax[1].set(title="Mean saved power", ylabel='mJ', xlabel='Time')

### Plot showing algorithm convergence time
sns.lineplot(x="datetime", y="convergence_time", data=data, ax=ax[2])
sns.lineplot(x="datetime", y=np.median(data["convergence_time"]), data=data, ax=ax[2], dashes=True)
ax[2].set(title="Algorithm convergence time", ylabel='s', xlabel='Time')

### Plot showing the correlation berween task latency and served mobile agents
p1 = sns.relplot(x="mean_task_latency", y="total_agents", data=data, ax=ax[3])
ax[3].set(title="Correlation between task latency and served mobile agents")
plt.close(p1.fig) # Workaround to eliminate double axis generated by relplot

### Plot showing the correlation between task latency and battery consumption
p2 = sns.relplot(x="mean_task_latency", y="mean_battery_usage", data=data, ax=ax[4])
ax[4].set(title="Correlation between task latency and battery consumption")
plt.close(p2.fig) # Workaround to eliminate double axis generated by relplot

### Plot showing the correlation between task latency and battery consumption
p3 = sns.relplot(x="total_agents", y="convergence_time", data=data, ax=ax[5])
ax[5].set(title="Correlation between number of agents and algorithm convergence time")
plt.close(p3.fig) # Workaround to eliminate double axis generated by relplot
ax[5].plot(
    np.unique(data['total_agents']),
    np.poly1d(np.polyfit(data['total_agents'],data['convergence_time'], 1))(np.unique(data['total_agents'])),
    'rx'
    )

fig.subplots_adjust(hspace=0.2)
fig.tight_layout()

# Saving plots
if hp.save_plots:
    fig.savefig("../plots/g1_plots.pdf")

############# PLOTS: GROUP 2 #############

fig, ax = plt.subplots(2, 1, figsize=(16, 12))
# CDF for task latency
powerlaw.plot_cdf(data=data["mean_task_latency"], ax=ax[0])
ax[0].set(title="CDF - Mean Task Latency")

# Relative load for each tier of edge nodes
ax[1].stackplot(
    data["datetime"].values,
    [data["t1-en-load"].values, data["t2-en-load"].values, data["t3-en-load"].values ],
    labels=['T1-EN','T2-EN','T3-EN']
    )
ax[1].legend(loc='upper left')
ax[1].set(title="EN utilization (detail)")
ax[1].set_yscale("log")

fig.subplots_adjust(hspace=0.2)
fig.tight_layout()

# Saving plots
if hp.save_plots:
    fig.savefig("../plots/g2_plots.pdf")

############# PLOTS: GROUP 3 #############

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Pie plot to show the percentage of EN usage for each tier
ta = data["total_agents"].sum()

v1 = np.sum(data["t1-en-load"])/ ta
v2 = np.sum(data["t2-en-load"])/ ta
v3 = np.sum(data["t3-en-load"])/ ta
v4 = data["cloud"].sum() / ta
v5 = data["local"].sum() / ta

ax[0].pie([v1, v2, v5, v3, v4], labels=['T1-EN','T2-EN','Local','T3-EN','Cloud'], autopct='%1.1f%%', shadow=True)
ax[0].set(title="EN utilization ratio")
ax[0].axis('equal')

# Pie plot to show the percentage of mobile agents saving battery
pos = len(data[data.mean_battery_usage > 0].mean_battery_usage) / len(data)
neg = len(data[data.mean_battery_usage < 0].mean_battery_usage) / len(data)

ax[1].pie([ pos, neg], labels=['Less','More'], autopct='%1.1f%%', shadow=True)
ax[1].set(title="Estimated battery utilization compared to local execution")
ax[1].axis('equal')

fig.subplots_adjust(hspace=0.2)
fig.tight_layout()

# Saving plots
if hp.save_plots:
    fig.savefig("../plots/g3_plots.pdf")

# %%
############# HEATMAP PLOT #############

# This doesn't work outside of Jupyter (apart from just saving the map in HTML)
import folium
from folium import plugins

building_users_density_heatmap = folium.Map(
    location=[48.150305,11.580054],
    tiles='stamentoner',
    zoom_start=12,
)

datas = []
flattened = [item for sublist in coord_dataframe_list for item in sublist]
for coord_list in flattened:
    st = coord_list.assign(norm_total=(coord_list.total/coord_list.total.max()))
    st.reset_index(inplace=True)
    datas.append(st[['timestamp', 'latitude', 'longitude', 'total']].values.tolist())

sl = []
for idx, elem in enumerate(datas[0]):
    temp = []
    for i in range(len(datas)):
        try:
            temp.append([datas[i][idx][1], datas[i][idx][2], datas[i][idx][3]])
        except IndexError:
            pass
    sl.append(temp)

hmt = plugins.HeatMapWithTime(sl,auto_play=True,use_local_extrema=True, max_opacity=0.8,index=flattened[0].reset_index().timestamp.tolist())
hmt.add_to(building_users_density_heatmap)

# building_users_density_heatmap
building_users_density_heatmap.save('../plots/building_users_density_heatmap.html')
# %%
