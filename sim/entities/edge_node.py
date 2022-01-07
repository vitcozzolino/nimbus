import numpy as np
import random
from config import hyperparams
import lib.inference_loader as inf
from entities.en_tier import Tier

class EdgeNode(object):
    def __init__(self, area, tier=Tier.One, _id=0, ap_name=None, mobile_agents_load=0):
        self.id = _id
        self.current_served_agents = []
        self.mobile_agents_load = mobile_agents_load
        self.tier = tier
        self.wifi_bw = 0
        self.cnn_exec_time = self.get_cnn_exec_time()
        self.model_accuracy = self.get_model_accuracy()
        self.location = self.get_location(area)
        
        self.worst_agent = None
        
        self.init_bw()
        self.ap_name = ap_name

    def init_bw(self):
        if self.tier == Tier.One or self.tier == Tier.AP:
            self.wifi_bw = np.random.choice(np.random.normal(300, 10, 100))
        elif self.tier == Tier.Two:
            self.wifi_bw = np.random.choice(np.random.normal(1000, 10, 100))
        else:
            self.wifi_bw = np.random.choice(np.random.normal(5000, 10, 100))
            
    def get_location(self, area):
        return (random.randint(0,area), random.randint(0,area))
    
    def get_max_servable_agents(self, max_rtt):
        return inf.get_max_agents_by_tier(max_rtt, self.tier)

    # This should be a list of models with accuracy
    def get_model_accuracy(self):
        return random.randint(60,80)

    def add_agent(self, agent):
        if self.worst_agent is None or (self.worst_agent.total_latency() < agent.total_latency()):
            self.worst_agent = agent

        self.current_served_agents.append(agent)

    def get_worst_agent(self):
        return self.worst_agent
    
    def get_cnn_exec_time(self):
        counter = len(self.current_served_agents) + 1
        comp = inf.get_compute_time_by_tier(self.tier)
        queue = inf.get_queue_time_by_tier(counter, self.tier) 
        tot = comp + (queue if queue > 0 else 0) 
        #print(queue)

        return tot

    def get_estimated_wifi_bandwidth(self):
        counter = self.mobile_agents_load + 1
        return self.wifi_bw/counter    
        
    def percentual_rtt_penalty(self):
        return 0
        # return (self.tier + 1)/10

    def p(self):
        return {i: self.__dict__ [i] for i in self.__dict__ if i!='current_served_agents'}
