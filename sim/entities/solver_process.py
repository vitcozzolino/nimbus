import threading
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from lib import solver
from lib.common import generate_agents, generate_edge_nodes
from entities.env_tracker import EnvTracker
import config.hyperparams as hp
from entities.en_tier import Tier
from entities.mobile_agent import Offloaded
from multiprocessing import Process, Queue

class SolverProcess(Process):
    def __init__(self, input_queue, result_queue, merged_raw_ap_data, peps, total_en, en_ratio, rtt_matrix, t1, t2, t3):
        super(SolverProcess, self).__init__()
        self.input_queue =input_queue
        self.result_queue = result_queue
        self.partial_episodes = peps
        self.total_en = total_en
        self.en_ratio = en_ratio
        self.rtt_matrix = rtt_matrix
        self.merged_raw_ap_data = merged_raw_ap_data
        self.tier1 = t1
        self.tier2 = t2
        self.tier3 = t3

        self.data = self.input_queue.get()

    def run(self):
        print("Starting solver process ...")
        for episode in tqdm(range(self.partial_episodes)):
            MA = int(self.data.iloc[episode].total)
            #env = EnvTracker(MA,self.total_en,self.partial_episodes)
            agents = generate_agents(MA, self.merged_raw_ap_data.loc[[self.data.iloc[episode].timestamp]], 0)
            edge_nodes = generate_edge_nodes(self.en_ratio, self.merged_raw_ap_data.loc[[self.data.iloc[episode].timestamp]], 0)

            env, agents, edge_nodes, conv_time, _ = \
                solver.algorithm_v3(
                    agents, edge_nodes, 5, 4, 100, episode,
                    None, self.rtt_matrix,
                    self.merged_raw_ap_data.loc[[self.data.iloc[episode].timestamp]],
                    suppress_output=True
                    )
            if hp.STORE_RESULTS:
                env.episodes_tracker[:,:,episode]
                np.savetxt(hp.CSV_FOLDER + '/' + str(episode) + '.csv', env.episodes_tracker[:,:,episode], delimiter=',')

            # Collect results
            saved_energy = list(map(lambda a: a.get_local_power_drain() - a.current_power_drain, agents))
            edge_nodes_associated_agents = list(map(lambda a: len(a.current_served_agents) ,edge_nodes))
            edge_nodes_tier = list(map(lambda a: a.tier ,edge_nodes))
            edge_node_bw = list(map(lambda a: np.round(a.get_estimated_wifi_bandwidth()) ,edge_nodes))

            d = {'Mobile Agents':edge_nodes_associated_agents,'Node Tier':edge_nodes_tier, 'Bandwidth':edge_node_bw}
            df = pd.DataFrame(d, columns=["Mobile Agents", "Node Tier", 'Bandwidth'])

            t = list(map(lambda a: 0 if a < 0 else a, [f.total_latency() for f in agents]))
            self.data.at[self.data.iloc[episode].timestamp, "mean_task_latency"] = np.mean(t)
            self.data.at[self.data.iloc[episode].timestamp, "max_task_latency"] = np.max(t)
            self.data.at[self.data.iloc[episode].timestamp, "min_task_latency"] = np.min(t)
            self.data.at[self.data.iloc[episode].timestamp, "std_task_latency"] = np.std(t)
            
            t = list(map(lambda a: 0 if a < 0 else a, [f.get_mobile_inference_latency() for f in agents]))
            self.data.at[self.data.iloc[episode].timestamp, "mean_local_inference_time"] = np.mean(t)
            self.data.at[self.data.iloc[episode].timestamp, "max_local_inference_time"] = np.max(t)
            self.data.at[self.data.iloc[episode].timestamp, "min_local_inference_time"] = np.min(t)
            self.data.at[self.data.iloc[episode].timestamp, "std_local_inference_time"] = np.std(t)

            self.data.at[self.data.iloc[episode].timestamp, "mean_battery_usage"] = np.mean(saved_energy)
            self.data.at[self.data.iloc[episode].timestamp, "max_battery_usage"] = np.max(saved_energy)
            self.data.at[self.data.iloc[episode].timestamp, "min_battery_usage"] = np.min(saved_energy)
            self.data.at[self.data.iloc[episode].timestamp, "std_battery_usage"] = np.std(saved_energy)

            self.data.at[self.data.iloc[episode].timestamp, "t1-en-load"] = \
                np.sum(df[df["Node Tier"] == Tier.One]["Mobile Agents"])
            self.data.at[self.data.iloc[episode].timestamp, "t2-en-load"] = \
                np.sum(df[df["Node Tier"] == Tier.Two]["Mobile Agents"])
            self.data.at[self.data.iloc[episode].timestamp, "t3-en-load"] = \
                np.sum(df[df["Node Tier"] == Tier.Three]["Mobile Agents"])
            self.data.at[self.data.iloc[episode].timestamp, "local"] = \
                len(list(filter(lambda a: a.offload_target == Offloaded.Local ,agents)))
            self.data.at[self.data.iloc[episode].timestamp, "cloud"] = \
                len(list(filter(lambda a: a.offload_target == Offloaded.Cloud ,agents)))
            self.data.at[self.data.iloc[episode].timestamp, "edge"] = \
                len(list(filter(lambda a: a.offload_target == Offloaded.Edge ,agents)))
            self.data.at[self.data.iloc[episode].timestamp, "convergence_time"] = conv_time
            self.data.at[self.data.iloc[episode].timestamp, "total_agents"] = MA

        print("Solver process completed!")
        self.result_queue.put(self.data)