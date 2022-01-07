import time
import numpy as np
from lib.common import generate_agents, generate_edge_nodes, locate_edge_nodes, get_estimated_latency
from lib.common import fair_allocation, get_estimated_tx_latency, calculate_power_drain
from lib import cloud_rtt_loader
from entities.mobile_agent import MobileAgent, Offloaded
from entities.edge_node import EdgeNode
from config import hyperparams
from tqdm import tqdm


# k only makes sense if we use a matrix to position agents and edge nodes
def algorithm(M = 10000, EN_RATIO = (100, 100, 100), dim=64, rangee=1, k=10, episode=0, env=None, rtt_matrix=None, merged_raw_ap_data=None, suppress_output=False):    
    agents = generate_agents(M, merged_raw_ap_data, dim)
    edge_nodes = generate_edge_nodes(EN_RATIO, merged_raw_ap_data, dim)
    c_rtt = cloud_rtt_loader.get_cloud_rtt()
    
    # hm1 = create_heatmap([f.location for f in agents], dim)
    # hm2 = create_heatmap([f.location for f in edge_nodes], dim, clr="bone")
    # ap = list(filter(lambda en: en.is_ap == True, edge_nodes))
    # hm3 = create_heatmap([f.location for f in ap], dim, clr="bone")

    excluded = []
    list_0 = []
    
    start = time.time()
    lte_counter = 0
    for agent in agents:
        list_0 = locate_edge_nodes(agent, edge_nodes, dim, rangee, k, rtt_matrix)
        # list_1 = filter_en_by_accuracy(agent, list_0)
        #print(list_1)
        #list_2 = filter_en_by_latency(agent, list_1)
        
        #list_0 = sort_en_by_load(list_0)
        
        # Here we need to assess the impact of scheduling agent_i on edge_node_j
        # print("{} - {}".format(idx, len(list_2)))
        #en_candidates = [offloaded[i] for i in flatten(list_2)]

        network_latency, inference_latency, least_loaded_wifi = fair_allocation(agent, list_0, connection="wifi", env=env, episode=episode)      
        lte_latency = get_estimated_tx_latency(agent, None, connection="lte") + np.random.choice(c_rtt)
        
        if env:
            env.update_cloud_latency(agent.id, lte_latency, episode=episode)
            env.update_local_latency(agent.id, agent.local_inference_time, episode=episode)
        
        lte_power_drain = calculate_power_drain(lte_latency, connection='lte')

        if (network_latency + inference_latency) > agent.max_RTT:
            if (lte_latency + hyperparams.cloud_inference) > agent.max_RTT:
                # Can't offload
                agent.current_power_drain = agent.get_local_power_drain()
                agent.inference_latency = agent.get_mobile_inference_latency()
                excluded.append(agent)
                if env:
                    env.update_agent_power_drain(agent.id, agent.get_local_power_drain(), episode)
                agent.offload_target = Offloaded.Local
        else:
            if (network_latency + inference_latency) <= (lte_latency + hyperparams.cloud_inference):
                #print("{}, {}".format(agent.get_local_power_drain(), calculate_power_drain(network_latency)))
                if calculate_power_drain(network_latency) <= agent.get_local_power_drain():
                    # Offload to edge node
                    least_loaded_wifi.add_agent(agent)
                    agent.offload_target = Offloaded.Edge
                    agent.network_latency = network_latency
                    for a in least_loaded_wifi.current_served_agents:
                        a.inference_latency = inference_latency
                        power_drain = calculate_power_drain(a.network_latency) 
                        a.current_power_drain = power_drain
                        
                        if env:
                            env.update_agent_power_drain(a.id, power_drain, episode)
            else:
                if lte_power_drain <= agent.get_local_power_drain():
                    # Offload to cloud
                    lte_counter += 1
                    agent.current_power_drain = lte_power_drain
                    agent.network_latency = lte_latency
                    agent.inference_latency = hyperparams.cloud_inference
                    agent.offload_target = Offloaded.Cloud
                    if env:
                        env.update_agent_power_drain(agent.id, lte_power_drain, episode)
            if env:
                env.update_agent_power_drain(agent.id, agent.get_local_power_drain(), episode)
                
    stop = time.time()
    convergence_time = stop - start

    if not suppress_output:
        # plot_power_hystogram([f.current_power_drain for f in agents])
        # plot_latency_hystogram([f.current_latency for f in agents])
        # plot_occupancy_heatmap(edge_nodes, dim)
        
        # plot_heatmap([hm1, hm2, hm3])
        # plot_bivariate_distr_power_latency(agents)
        # plot_saved_power_hystogram(agents)
        # plot_occupancy_hystogram(edge_nodes)
        
        print("=== PARAMETERS ===") 
        print("Mobile agents: {}".format(M))
        print("Edge nodes: {}".format(EN_RATIO))
        print("Grid size: {}x{}".format(dim,dim))
        print("Max concurrency: 64")
        print("Quadrant search radius: {}".format(rangee))
        print("Clustering factor: {}".format(k))

        print("=== RESULTS ===")
        print("LTE Offloaded: {}".format(lte_counter))
        #for cls in hyperparams.DeviceLoad:
        #    tot = len(list(filter(lambda en: en.current_load_class == cls, edge_nodes)))
        #    print("Nodes in {} state: {}".format(cls, tot))            

        print("Discarded mobile agents {}".format(len(excluded)))
        overbooked_en = len([en for en in edge_nodes if len(en.current_served_agents) > en.max_servable_agents])
        print("Total overbooked EN: {}".format(overbooked_en))
        print("Mean network bandwidth: {} Mbps".format(np.mean([en.wifi_bw for en in edge_nodes])))
        print("Done in {} seconds".format(convergence_time))
    
    return env, agents, edge_nodes, convergence_time, excluded


def algorithm_v2(agents, edge_nodes, dim=64, rangee=1, k=10, episode=0, env=None, rtt_matrix=None, merged_raw_ap_data=None, suppress_output=False):    
    c_rtt = cloud_rtt_loader.get_cloud_rtt()

    excluded = []
    list_0 = []
    
    start = time.time()
    lte_counter = 0
    for agent in agents:
        list_0 = locate_edge_nodes(agent, edge_nodes, dim, rangee, k, rtt_matrix)

        network_latency, inference_latency, least_loaded_wifi = fair_allocation(agent, list_0, connection="wifi", env=env, episode=episode)      
        lte_latency = get_estimated_tx_latency(agent, None, connection="lte") + np.random.choice(c_rtt)
        
        if env:
            env.update_cloud_latency(agent.id, lte_latency, episode=episode)
            env.update_local_latency(agent.id, agent.local_inference_time, episode=episode)
        
        lte_power_drain = calculate_power_drain(lte_latency, connection='lte')

        if (network_latency + inference_latency) > agent.max_RTT:
            if (lte_latency + hyperparams.cloud_inference) > agent.max_RTT:
                # Can't offload
                agent.current_power_drain = agent.get_local_power_drain()
                agent.inference_latency = agent.get_mobile_inference_latency()
                excluded.append(agent)
                if env:
                    env.update_agent_power_drain(agent.id, agent.get_local_power_drain(), episode)
                agent.offload_target = Offloaded.Local
        else:
            if (network_latency + inference_latency) <= (lte_latency + hyperparams.cloud_inference):
                #print("{}, {}".format(agent.get_local_power_drain(), calculate_power_drain(network_latency)))
                if calculate_power_drain(network_latency) <= agent.get_local_power_drain():
                    # Offload to edge node
                    least_loaded_wifi.add_agent(agent)
                    agent.offload_target = Offloaded.Edge
                    agent.network_latency = network_latency
                    for a in least_loaded_wifi.current_served_agents:
                        a.inference_latency = inference_latency
                        power_drain = calculate_power_drain(a.network_latency) 
                        a.current_power_drain = power_drain
                        
                        if env:
                            env.update_agent_power_drain(a.id, power_drain, episode)
            else:
                if lte_power_drain <= agent.get_local_power_drain():
                    # Offload to cloud
                    lte_counter += 1
                    agent.current_power_drain = lte_power_drain
                    agent.network_latency = lte_latency
                    agent.inference_latency = hyperparams.cloud_inference
                    agent.offload_target = Offloaded.Cloud
                    if env:
                        env.update_agent_power_drain(agent.id, lte_power_drain, episode)
                
    stop = time.time()
    convergence_time = stop - start
    
    return env, agents, edge_nodes, convergence_time, excluded


def algorithm_v3(agents, edge_nodes, dim=64, rangee=1, k=10, episode=0, env=None, rtt_matrix=None, merged_raw_ap_data=None, suppress_output=False, connection='wifi'):    
    c_rtt = cloud_rtt_loader.get_cloud_rtt()

    excluded = []
    list_0 = []
    
    start = time.time()
    lte_counter = 0
    for agent in agents:
        list_0 = locate_edge_nodes(agent, edge_nodes, dim, rangee, k, rtt_matrix)

        network_latency, inference_latency, least_loaded_wifi = fair_allocation(agent, list_0, connection="wifi", env=env, episode=episode)      
        
        if connection == 'wifi':
            edge_node = list(filter(lambda en: en.ap_name == agent.ap_name, edge_nodes))[0]
            lte_latency = get_estimated_tx_latency(agent, edge_node, connection) + 1.2*abs(np.random.choice(c_rtt))
            lte_power_drain = calculate_power_drain(lte_latency, connection)
        else:
            lte_latency = get_estimated_tx_latency(agent, None, connection) + abs(np.random.choice(c_rtt))
            lte_power_drain = calculate_power_drain(lte_latency, connection)
        
        if env:
            env.update_cloud_latency(agent.id, lte_latency, episode=episode)
            env.update_local_latency(agent.id, agent.local_inference_time, episode=episode)

        if (network_latency + inference_latency) > agent.max_RTT:
            if (lte_latency + hyperparams.cloud_inference) > agent.max_RTT:
                # Can't offload
                agent.current_power_drain = agent.get_local_power_drain()
                agent.inference_latency = agent.get_mobile_inference_latency()
                excluded.append(agent)
                if env:
                    env.update_agent_power_drain(agent.id, agent.get_local_power_drain(), episode)
                agent.offload_target = Offloaded.Local
        else:
            if (network_latency + inference_latency) <= (lte_latency + hyperparams.cloud_inference):
                #print("{}, {}".format(agent.get_local_power_drain(), calculate_power_drain(network_latency)))
                if calculate_power_drain(network_latency) <= agent.get_local_power_drain():
                    # Offload to edge node
                    least_loaded_wifi.add_agent(agent)
                    agent.offload_target = Offloaded.Edge
                    agent.network_latency = network_latency
                    for a in least_loaded_wifi.current_served_agents:
                        a.inference_latency = inference_latency
                        power_drain = calculate_power_drain(a.network_latency) 
                        a.current_power_drain = power_drain
                        
                        if env:
                            env.update_agent_power_drain(a.id, power_drain, episode)
            else:
                if lte_power_drain <= agent.get_local_power_drain():
                    # Offload to cloud
                    lte_counter += 1
                    agent.current_power_drain = lte_power_drain
                    agent.network_latency = lte_latency
                    agent.inference_latency = hyperparams.cloud_inference
                    agent.offload_target = Offloaded.Cloud
                    excluded.append(agent)
                    if env:
                        env.update_agent_power_drain(agent.id, lte_power_drain, episode)
                
    stop = time.time()
    convergence_time = stop - start
    
    return env, agents, edge_nodes, convergence_time, excluded

def algorithm_cloud(agents, edge_nodes, dim=64, rangee=1, k=10, episode=0, env=None, rtt_matrix=None, merged_raw_ap_data=None, suppress_output=False, connection='lte'):    
    c_rtt = cloud_rtt_loader.get_cloud_rtt()
    
    start = time.time()
    excluded = []

    for agent in agents:
        if connection == 'wifi':
            edge_node = list(filter(lambda en: en.ap_name == agent.ap_name, edge_nodes))[0]
            lte_latency = get_estimated_tx_latency(agent, edge_node, connection) + 1.2*abs(np.random.choice(c_rtt))
            lte_power_drain = calculate_power_drain(lte_latency, connection)
        else:
            lte_latency = get_estimated_tx_latency(agent, None, connection) + np.random.choice(c_rtt)
            lte_power_drain = calculate_power_drain(lte_latency, connection)

        if env:
            env.update_cloud_latency(agent.id, lte_latency, episode=episode)
            env.update_local_latency(agent.id, agent.local_inference_time, episode=episode)
        
        if (lte_latency + hyperparams.cloud_inference) < agent.max_RTT and lte_power_drain <= agent.get_local_power_drain():
            agent.current_power_drain = lte_power_drain
            agent.network_latency = lte_latency
            agent.inference_latency = hyperparams.cloud_inference
            agent.offload_target = Offloaded.Cloud
            if env:
                env.update_agent_power_drain(agent.id, lte_power_drain, episode)
            if env:
                env.update_agent_power_drain(agent.id, agent.get_local_power_drain(), episode)
        else:
            agent.current_power_drain = agent.get_local_power_drain()
            agent.inference_latency = agent.get_mobile_inference_latency()
            excluded.append(agent)
            
            if env:
                env.update_agent_power_drain(agent.id, agent.get_local_power_drain(), episode)
            
            agent.offload_target = Offloaded.Local

    stop = time.time()
    convergence_time = stop - start
    
    return env, agents, edge_nodes, convergence_time, excluded

def algorithm_p2(agents, edge_nodes, dim=64, rangee=1, k=10, episode=0, env=None, rtt_matrix=None, merged_raw_ap_data=None, suppress_output=False, connection='wifi'):    
    # P2 version of the algorithm where we just try to place more nodes at the edge after executing the distributed phase
    excluded = []
    list_0 = []
    
    start = time.time()

    for agent in agents:
        list_0 = locate_edge_nodes(agent, edge_nodes, dim, rangee, k, rtt_matrix)

        network_latency, inference_latency, least_loaded_wifi = fair_allocation(agent, list_0, connection="wifi", env=env, episode=episode)      
        
        if (network_latency + inference_latency) < (agent.network_latency + agent.inference_latency):
            if calculate_power_drain(network_latency) <= agent.get_local_power_drain():
                # Offload to edge node
                least_loaded_wifi.add_agent(agent)
                agent.offload_target = Offloaded.Edge
                agent.network_latency = network_latency
                for a in least_loaded_wifi.current_served_agents:
                    a.inference_latency = inference_latency
                    power_drain = calculate_power_drain(a.network_latency) 
                    a.current_power_drain = power_drain
                    
                    if env:
                        env.update_agent_power_drain(a.id, power_drain, episode)
                
    stop = time.time()
    convergence_time = stop - start

    return env, agents, edge_nodes, convergence_time, excluded