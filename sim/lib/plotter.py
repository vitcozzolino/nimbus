import pandas as pd
import matplotlib, math, gc, itertools
import numpy as np
import matplotlib.pyplot as plt
import glob, os, platform, re, time
import numpy as np
import multiprocessing
from collections import Counter
import seaborn as sns
#import powerlaw
from scipy.stats import mannwhitneyu
from importlib import reload

import sys

# plt.style.use('classic')

matplotlib.rcParams['axes.facecolor'] = 'w'
matplotlib.rcParams['axes.edgecolor'] = 'k'
matplotlib.rcParams['figure.facecolor'] = 'w'
matplotlib.rcParams['axes.titlesize'] = 22
matplotlib.rcParams['axes.labelsize'] = 22
matplotlib.rcParams['lines.linewidth'] = 2
matplotlib.rcParams['lines.markersize'] = 3
matplotlib.rcParams['lines.color'] = 'xkcd:blue'
matplotlib.rcParams['xtick.labelsize'] = 20
matplotlib.rcParams['ytick.labelsize'] = 20

# width = 7
# height = width / 1.618

def create_heatmap(matrix, dim, clr='hot'):
    hm = np.zeros([dim + 1, dim + 1])
    for point in matrix:
        hm[point[0], point[1]] = hm[point[0], point[1]] + 1

    return hm

def plot_heatmap(hm: list, _cmap='Oranges'):
    total_fig = len(hm)
    fig, ax = plt.subplots(1, total_fig, figsize=(12, 4))

    for i in range(total_fig):
        sns.heatmap(hm[i], cmap=_cmap, ax=ax[i] )
        ax[i].axis('off')
    
    fig.tight_layout(w_pad=1.5)

    
def plot_bivariate_distr_power_latency(agents):
    try:
        pwr = list(map(lambda a: 0 if a < 0 else a, [f.current_power_drain for f in agents]))
        lat = list(map(lambda a: 0 if a < 0 else a, [f.current_latency for f in agents]))
        target = list(map(lambda a: a.offload_target, agents))
        payload = list(map(lambda a: a.payload, agents))

        d = {'Power':pwr,'Task Execution':lat, 'Execution Platform':target}
        df = pd.DataFrame(d, columns=["Power", "Task Execution", "Execution Platform"])

        fig = plt.figure(figsize=(8,8))
    #     sns.jointplot(x="Power", y="Task Execution", hue="Execution Platform", data=df)
    #     plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    #     plt.axis('off')


        g = sns.JointGrid("Power", "Task Execution", df)
        for var1, var2 in df.groupby("Execution Platform"):
            sns.kdeplot(var2["Power"], ax=g.ax_marg_x, legend=False)
            sns.kdeplot(var2["Task Execution"], ax=g.ax_marg_y, vertical=True, legend=False)
            g.ax_joint.plot(var2["Power"], var2["Task Execution"], "o", ms=5, label=var1)

        legend_properties = {'weight':'bold','size':8}
        legendMain=g.ax_joint.legend(prop=legend_properties,loc='upper right')

        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.axis('off')
    except:
        print("Soemthing went wrong when plotting.")
        pass
    
def plot_power_hystogram(agents_power_drain):
    try:
        plt.hist(list(filter(lambda a: a > 0, agents_power_drain)), 10, density=False, alpha=0.75)
        plt.show()
    except:
        print("Soemthing went wrong when plotting.")
        pass
    
def plot_saved_power_hystogram(agents):
    try:
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))

        data = []
        for a in agents:
            data.append(a.current_power_drain - a.local_exec_power_drain * a.local_inference_time)
    #     sns.distplot(data, bins=20)

        cpd = list(map(lambda a: a.current_power_drain, agents))
        lpd = list(map(lambda a: a.local_exec_power_drain * a.local_inference_time, agents))

        d = {'Current Power':cpd,'Local Power':lpd}
        df = pd.DataFrame(d, columns=["Current Power", "Local Power"])

        sns.kdeplot(df["Current Power"], shade=1, color='red', ax=ax[0])
        sns.kdeplot(df["Local Power"], shade=1, color='green', ax=ax[0])

        payload = list(map(lambda a: a.payload, agents))
        sns.scatterplot(np.linspace(0, len(agents), num=len(agents)), payload, ax=ax[1])

        plt.xlim(left=0)
    except:
        print("Soemthing went wrong when plotting.")
        pass
#     plt.hist(data, density=True, cumulative=True, histtype='step', alpha=0.75)
#     plt.show()
    
def plot_latency_hystogram(agents):
    try:
        curr_lat = list(filter(lambda a: a.current_latency > 0, agents))
        plt.hist(curr_lat, 10, density=False, alpha=0.75)
        plt.show()
    except:
        print("Soemthing went wrong when plotting.")
        pass

def plot_occupancy_hystogram(edge_nodes):
    try:
        fig = plt.figure(figsize=(16, 8))
        edge_nodes_associated_agents = list(map(lambda a: len(a.current_served_agents) ,edge_nodes))
        edge_nodes_tier = list(map(lambda a: a.tier ,edge_nodes))
        edge_node_bw = list(map(lambda a: np.round(a.get_estimated_wifi_bandwidth()) ,edge_nodes))

        x = np.array(edge_node_bw)
        bins = np.array([15, 30, 45, 60, 75, 90])
        binned_edge_node_bw = np.digitize(x,bins,right=True)


        d = {'Mobile Agents':edge_nodes_associated_agents,'Node Tier':edge_nodes_tier, 'Bandwidth':binned_edge_node_bw}
        df = pd.DataFrame(d, columns=["Mobile Agents", "Node Tier", 'Bandwidth'])

        sns.violinplot(x='Node Tier', y="Mobile Agents", palette="Set3", data=df, inner=None)
        sns.swarmplot(
            x='Node Tier',
            y='Mobile Agents',
            data=df,
            hue=df['Bandwidth'].values
        )
    except:
        print("Soemthing went wrong when plotting.")
        pass
    

def plot_occupancy_heatmap(edge_nodes, dim, clr='hot'):
    a = np.zeros([dim + 1, dim + 1])
    for f in edge_nodes:
        a[f.location[0], f.location[1]] = len(f.current_served_agents)

    plt.imshow(a, cmap=clr, interpolation='nearest')
    plt.show()
    