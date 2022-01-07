import threading
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from lib import solver
from entities.env_tracker import EnvTracker
import config.hyperparams as hp
from entities.en_tier import Tier
from entities.mobile_agent import Offloaded
from multiprocessing import Process, Queue

class SolverProcessNS(Process):
    def __init__(self, result_queue, agents, edge_nodes, rtt_matrix, merged_raw_ap_data, cloud_only=False):
        super(SolverProcessNS, self).__init__()
        self.result_queue = result_queue
        self.agents = agents
        self.edge_nodes = edge_nodes
        self.rtt_matrix = rtt_matrix
        self.merged_raw_ap_data = merged_raw_ap_data
        self.cloud_only = cloud_only

    def run(self):
        if self.cloud_only:
            env, agents_a, edge_nodes_e, convergence_time, excluded = \
                solver.algorithm_cloud(self.agents, self.edge_nodes, 5, 4, 100, None, None, self.rtt_matrix, self.merged_raw_ap_data, suppress_output=True)
        else:
            env, agents_a, edge_nodes_e, convergence_time, excluded = \
                solver.algorithm_v3(self.agents, self.edge_nodes, 5, 4, 100, None, None, self.rtt_matrix, self.merged_raw_ap_data, suppress_output=True)

        self.result_queue.put((env, agents_a, edge_nodes_e, convergence_time, excluded))