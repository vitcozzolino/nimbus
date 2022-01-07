import numpy as np

class EnvTracker:
    def __init__(self, agents, edge_nodes, episodes=1):
        self.total_agents = agents
        self.total_edge_nodes = edge_nodes
        self.total_episodes = episodes
        self.episodes_tracker = np.full((self.total_agents, self.total_edge_nodes + 1 + 1 + 1, self.total_episodes), np.inf)
        
    def update_agent(self, agent_idx, en_idx, value, episode):
        self.episodes_tracker[agent_idx, en_idx, episode] = value
        
    def update_agent_power_drain(self, agent_idx, power_drain, episode):
        self.update_agent(agent_idx, self.total_edge_nodes + 2, power_drain, episode)
    
    def update_cloud_latency(self, agent_idx, latency, episode):
        self.update_agent(agent_idx, self.total_edge_nodes, latency, episode)
        
    def update_local_latency(self, agent_idx, latency, episode):
        self.update_agent(agent_idx, self.total_edge_nodes + 1, latency, episode)