#%%
import os, time
import numpy as np
from tqdm import tqdm
from lib import solver
from entities.env_tracker import EnvTracker
from entities.mobile_agent import Offloaded
from lib import wlanlrz_loader
from lib import rtt_matrix_loader 
from lib.plotter import *
import powerlaw
import matplotlib.pyplot as plt
import seaborn as sns
import config.hyperparams as hp
import datetime as dt

### WLAN-LRZ AP Data Loader ####
desc = wlanlrz_loader.load_data_description(hp.AP_DESCR)
data, total_ap = wlanlrz_loader.load_data(hp.BUILDING, desc)

print("Total APs: {}".format(total_ap))
data['timestamp'] = data.index

# Filter the dataser based on the minimum amount of users we want to serve
data = data[data.total >= hp.minimum_agents_threshold]

TOTAL_EPISODES = len(data)
TIER_1_EN = round(total_ap)
TIER_2_EN = round(total_ap/5)
TIER_3_EN = 1 #round(total_ap/30)
EN_RATIO = (TIER_1_EN, TIER_2_EN, TIER_3_EN)
TOTAL_EN = TIER_1_EN + TIER_2_EN + TIER_3_EN

if hp.STORE_RESULTS:
    try:
        os.mkdir(hp.CSV_FOLDER)
    except OSError:
        print ("Creation of the directory %s failed" % hp.CSV_FOLDER)
    else:
        print ("Successfully created the directory %s " % hp.CSV_FOLDER)

if hp.use_dataset_rtt:
    ### LOAD LATENCY MATRIX FROM DATASET ###
    print("Loading rtt matrix from dataset")
    clf = rtt_matrix_loader.analyze_data(source="PlanetLab", drange=1000, threshold=250, k=3)
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

# Manually setting latency for colocated AP/EN
for i in range(TIER_1_EN):
    for j in range(TIER_1_EN):
        if i == j:
            rtt_matrix[i][j] = 0.2
            
data["mean_task_latency"] = np.nan
data["mean_battery_usage"] = np.nan
data["convergence_time"] = np.nan

for episode in tqdm(range(TOTAL_EPISODES)):
    # Run the solver
    # M = 10000, N = 300, dim=64, rangee=1, k=10, episode=0
    MA = int(data.iloc[episode].total)
    env = EnvTracker(MA,TOTAL_EN,TOTAL_EPISODES)
    env, agents, edge_nodes, conv_time = solver.algorithm(MA, EN_RATIO, 5, 4, 100, episode, env, rtt_matrix, suppress_output=True)
    if hp.STORE_RESULTS:
        env.episodes_tracker[:,:,episode]
        np.savetxt(hp.CSV_FOLDER + '/' + str(episode) + '.csv', env.episodes_tracker[:,:,episode], delimiter=',')

    # Collect results
    saved_energy = list(map(lambda a: (a.local_exec_power_drain * a.local_inference_time) - a.current_power_drain, agents))
    edge_nodes_associated_agents = list(map(lambda a: len(a.current_served_agents) ,edge_nodes))
    edge_nodes_tier = list(map(lambda a: a.tier ,edge_nodes))
    edge_node_bw = list(map(lambda a: np.round(a.get_estimated_wifi_bandwidth()) ,edge_nodes))

    d = {'Mobile Agents':edge_nodes_associated_agents,'Node Tier':edge_nodes_tier, 'Bandwidth':edge_node_bw}
    df = pd.DataFrame(d, columns=["Mobile Agents", "Node Tier", 'Bandwidth'])

    data.at[data.iloc[episode].timestamp, "mean_task_latency"] = np.mean(list(map(lambda a: 0 if a < 0 else a, [f.current_latency for f in agents])))
    data.at[data.iloc[episode].timestamp, "mean_battery_usage"] = np.mean(saved_energy)
    data.at[data.iloc[episode].timestamp, "t1-en-load"] = np.sum(df[df["Node Tier"] == "small_EN"]["Mobile Agents"])/(TIER_1_EN * hp.HYPERPARAMS["standard_solver"]["small_EN"]["max_servable_agents"])
    data.at[data.iloc[episode].timestamp, "t2-en-load"] = np.sum(df[df["Node Tier"] == "medium_EN"]["Mobile Agents"])/(TIER_2_EN * hp.HYPERPARAMS["standard_solver"]["medium_EN"]["max_servable_agents"])
    data.at[data.iloc[episode].timestamp, "t3-en-load"] = np.sum(df[df["Node Tier"] == "big_EN"]["Mobile Agents"])/(TIER_3_EN * hp.HYPERPARAMS["standard_solver"]["big_EN"]["max_servable_agents"])
    data.at[data.iloc[episode].timestamp, "total_agents"] = MA
    data.at[data.iloc[episode].timestamp, "local"] = len(list(filter(lambda a: a.offload_target == Offloaded.Local ,agents)))
    data.at[data.iloc[episode].timestamp, "cloud"] = len(list(filter(lambda a: a.offload_target == Offloaded.Cloud ,agents)))
    data.at[data.iloc[episode].timestamp, "edge"] = len(list(filter(lambda a: a.offload_target == Offloaded.Edge ,agents)))
    data.at[data.iloc[episode].timestamp, "convergence_time"] = conv_time

# %%
data["datetime"] = pd.to_datetime(data['timestamp'], unit='s')

############# PLOTS: GROUP 1 #############

fig, ax = plt.subplots(4, 1, figsize=(16, 16))

# Plot showing the mean task execution time for each agent
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

# Plot showing the mean battery saving for each agent compared to running locally
sns.lineplot(x="datetime", y="mean_battery_usage", data=data, ax=ax[1])
sns.lineplot(x="datetime", y=np.median(data["mean_battery_usage"]), data=data, ax=ax[1], dashes=True)
ax[1].set(title="Mean saved battery", ylabel='Mw/h', xlabel='Time')

# Plot showing algorithm convergence time
sns.lineplot(x="datetime", y="convergence_time", data=data, ax=ax[2])
sns.lineplot(x="datetime", y=np.median(data["convergence_time"]), data=data, ax=ax[2], dashes=True)
ax[2].set(title="Algorithm convergence time", ylabel='s', xlabel='Time')

p1 = sns.relplot(x="mean_task_latency", y="total_agents", data=data, ax=ax[3])
plt.close(p1.fig) # Workaround to eliminate double axis generated by relplot

fig.subplots_adjust(hspace=0.2)
fig.tight_layout()

############# PLOTS: GROUP 2 #############

fig, ax = plt.subplots(3, 1, figsize=(16, 16))
# CDF for task latency
powerlaw.plot_cdf(data=data["mean_task_latency"], ax=ax[0])

# Relative load for each tier of edge nodes
ax[1].stackplot(
    data["datetime"].values,
    [data["t1-en-load"].values, data["t2-en-load"].values, data["t3-en-load"].values ],
    labels=['T1-EN','T2-EN','T3-EN']
    )
ax[1].legend(loc='upper left')
ax[1].set_yscale("log")

ta = data["total_agents"].sum()

v1 = np.sum(data["t1-en-load"]) * (TIER_1_EN * hp.HYPERPARAMS["standard_solver"]["small_EN"]["max_servable_agents"]) / ta
v2 = np.sum(data["t2-en-load"]) * (TIER_2_EN * hp.HYPERPARAMS["standard_solver"]["medium_EN"]["max_servable_agents"])/ ta
v3 = np.sum(data["t3-en-load"]) * (TIER_3_EN * hp.HYPERPARAMS["standard_solver"]["big_EN"]["max_servable_agents"])/ ta
v4 = data["cloud"].sum() / ta
v5 = data["local"].sum() / ta
# Pie plot to show the percentage of EN usage for each tier
ax[2].pie([v1, v2, v5, v3, v4], labels=['T1-EN','T2-EN','Local','T3-EN','Cloud'], autopct='%1.1f%%', shadow=True)
ax[2].axis('equal')

fig.subplots_adjust(hspace=0.2)
fig.tight_layout()

############# PLOTS: GROUP 3 #############

# This plots will only show value collected in the last iteration of the solver
plot_bivariate_distr_power_latency(agents)
plot_saved_power_hystogram(agents)
plot_occupancy_hystogram(edge_nodes)

fig, ax = plt.subplots(1, 2, figsize=(16, 8))
sns.distplot(rtt_matrix.flatten(), hist=True, rug=True, bins=50, ax=ax[0])
sns.heatmap(rtt_matrix, ax=ax[1])

#powerlaw.plot_cdf(rtt_matrix.flatten(), ax=ax[0])

# %%
