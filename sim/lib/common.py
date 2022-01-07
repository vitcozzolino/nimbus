import time
import math
import random
import numpy as np
import sys
from config import hyperparams

from entities.edge_node import EdgeNode
from entities.en_tier import Tier
from entities.mobile_agent import MobileAgent

def disarrange(a, axis=-1):
    """
    Shuffle `a` in-place along the given axis.

    Apply numpy.random.shuffle to the given axis of `a`.
    Each one-dimensional slice is shuffled independently.
    """
    b = a.swapaxes(axis, -1)
    # Shuffle `b` in-place along the last axis.  `b` is a view of `a`,
    # so `a` is shuffled in place, too.
    shp = b.shape[:-1]
    for ndx in np.ndindex(shp):
        np.random.shuffle(b[ndx])
    return

def calculateDistance(agent, edge_node):  
    dist = math.sqrt((agent.location[0] - edge_node.location[0])**2 + (agent.location[1] - edge_node.location[1])**2)  
    return dist

def get_estimated_latency():
    return random.randint(15, 40)

# TX latency in ms
def get_estimated_tx_latency(agent, edge_node, connection='wifi'):
    bw = edge_node.get_estimated_wifi_bandwidth() if connection is 'wifi' else agent.get_estimated_lte_bandwidth()
    lt = ((agent.payload/1000)/(bw/8)) * 10**3
    #print(lt)
    return lt

### Filtering functions (constraints) ###

## LEGACY CODE ##
# def filter_en_by_range(agent, edge_nodes, dim, rangee=1, k=10):
#     quadrant_rows_range = list(filter(lambda v: v >= 0 and v <= dim, range(agent.location[0] - rangee, agent.location[0] + rangee)))
#     quadrant_cols_range = list(filter(lambda v: v >= 0 and v <= dim, range(agent.location[1] - rangee, agent.location[1] + rangee)))
    
#     edge_nodes_in_quadrant = list(filter(
#         lambda v:
#             v.location[0] >= quadrant_rows_range[0] and v.location[0] <= quadrant_rows_range[-1]
#             and v.location[1] >= quadrant_cols_range[0] and v.location[1] <= quadrant_cols_range[-1], edge_nodes))
    
#     # Excluded overlaoded agents
#     non_overbooked_edge_nodes_in_quadrant = list(filter(lambda en: len(en.current_served_agents) < en.max_servable_agents, edge_nodes_in_quadrant))
    
#     # Combine distance with edge node
#     dist_node_list = list(map(lambda en: (calculateDistance(agent, en), en), non_overbooked_edge_nodes_in_quadrant))
    
#     # Discard EN out of range
#     in_range_en = list(filter(lambda en: en[0] <= rangee, dist_node_list))
    
#     # Sort by range
#     sorted_by_range = sorted(in_range_en, key= lambda x: x[0])
    
#     return sorted_by_range

def locate_edge_nodes(agent, edge_nodes, dim, rangee=1, k=0, rtt_matrix=None):
    # quadrant_rows_range = list(filter(lambda v: v >= 0 and v <= dim, range(agent.location[0] - rangee, agent.location[0] + rangee)))
    # quadrant_cols_range = list(filter(lambda v: v >= 0 and v <= dim, range(agent.location[1] - rangee, agent.location[1] + rangee)))
    
    # ap_in_quadrant = list(filter(
    #     lambda v:
    #         v.is_ap and
    #         v.location[0] >= quadrant_rows_range[0] and v.location[0] <= quadrant_rows_range[-1]
    #         and v.location[1] >= quadrant_cols_range[0] and v.location[1] <= quadrant_cols_range[-1], edge_nodes))
      
    # Discard AP out of range
    # in_range_ap = list(filter(lambda ap: calculateDistance(agent, ap) <= rangee, ap_in_quadrant))

    # Gets all the EN which are AP. Then, it remove the ones that are have been already served.
    ap_id = agent.ap_id
    
    #en_ap = list(filter(lambda ap: ap.mobile_agents_load - len(ap.current_served_agents) > 0, edge_nodes))[0]

    # Return RTT for each EN
    #rtt, ap_id = get_en_rtt(en_ap, rtt_matrix)            

    rtt = rtt_matrix[ap_id, :]

    # Get AP
    en_ap = list(filter(lambda en: en.ap_name == agent.ap_name, edge_nodes))[0]

    # Get all EN and exclude AP only
    edge_nodes = list(filter(lambda en: en.tier != Tier.AP, edge_nodes)) 

    # Build a new list of tuples with [en, rtt] but filter out RTT values higher than the max RTT for the agent
    rtt_en_list = []
    best_en_greedy = []

    best_rtt_greedy = np.inf
    for en, rtt_v in zip(edge_nodes, rtt):
        if agent.max_RTT >= rtt_v:
            rtt_en_list.append((en, rtt_v, en_ap))
            
            if hyperparams.greedy and rtt_v < best_rtt_greedy:
                best_en_greedy = [(en, rtt_v, en_ap)]
                best_rtt_greedy = rtt_v

    # Identify non-overloaded EN
    compatible_edge_nodes = []
    
    if hyperparams.greedy == False:
        for elem in rtt_en_list:
            en = elem[0]
            worst_agent = en.get_worst_agent()

            if worst_agent:
                worst_agent_net_latency = worst_agent.network_latency

                if worst_agent_net_latency + en.get_cnn_exec_time() <= worst_agent.max_RTT:
                    compatible_edge_nodes.append((en, elem[1], elem[2]))
            else:
                compatible_edge_nodes.append((en, elem[1], elem[2]))
    else:
        compatible_edge_nodes = best_en_greedy
    
    if hyperparams.local_search_scope == 0 or len(compatible_edge_nodes) == 0:
        return compatible_edge_nodes
    else:
        return random.sample(compatible_edge_nodes, np.minimum(hyperparams.local_search_scope, len(compatible_edge_nodes)))

                    
def filter_en_by_accuracy(agent, edge_nodes):
    s = list(filter(lambda en: agent.required_accuracy <= en.model_accuracy, edge_nodes))
    return sorted(s, key= lambda x: x.model_accuracy, reverse=True)

def filter_en_by_latency(agent, edge_nodes):
    return list(filter(lambda en: agent.max_RTT >= get_estimated_tx_latency(agent, en) + en.get_cnn_exec_time(), edge_nodes))

def sort_en_by_load(edge_nodes):
    return sorted(edge_nodes, key= lambda x: len(x.current_served_agents), reverse=True)

def get_en_rtt(ap_list, rtt_matrix):
    temp = []
                    
    # Collect RTTs for ENs accessible through the APs in range
    for i in ap_list:
        temp.append(rtt_matrix[i.id,:])           
    
    if temp:
        rtt = np.vstack(temp)
        min_rtt_per_en = np.amin(rtt, axis=0) # For each EN, get the smallest RTT
        selected_ap = [ap_list[i] for i in np.argmin(rtt, axis=0)]
    else:
        return [], []

    return min_rtt_per_en, selected_ap

def fair_allocation(agent, edge_nodes, connection="wifi", env=None, episode=0):
    alpha = hyperparams.alpha
    beta = hyperparams.beta
    
    temp = sys.maxsize
    
    tgt_node = None
    tgt_network_latency = np.inf
    tgt_inference_latency = np.inf

    # Hill climbing local search algorithm variant
    for elem in edge_nodes:
        en = elem[0]
        rtt = elem[1]
        ap = elem[2]

        min_bw = np.min([get_estimated_tx_latency(agent, ap, connection), get_estimated_tx_latency(agent, en, connection)])  

        network_latency = min_bw + rtt*(1 + en.percentual_rtt_penalty())
        inference_latency = en.get_cnn_exec_time()
        
        latency = network_latency + inference_latency

        objective = \
            alpha * (latency/agent.max_RTT) + beta * (len(en.current_served_agents)/en.get_max_servable_agents(agent.max_RTT))
        
        # Update the env snapshot
        if env is not None:
            env.update_agent(agent.id, en.id, latency, episode)
        
        if objective < temp:
            temp = objective
            tgt_network_latency = network_latency
            tgt_inference_latency = inference_latency
            tgt_node = en
            
    return tgt_network_latency, tgt_inference_latency, tgt_node

# def calculate_power_drain(agent, edge_node, connection="wifi"):
#     alpha_u = 283.17 if connection == "wifi" else 438.39
#     beta_u = 132.86 if connection == "wifi" else 1288.04
    
#     P = alpha_u * edge_node.get_estimated_wifi_bandwidth() + beta_u - agent.local_exec_power_drain
#     agent.current_power_drain = P

#General model Power2:     f(x) = a*x^b+c
# def calculate_power_drain(agent, connection="wifi"):
#     a = 4.733 if connection is "wifi" else 9.549
#     b = -0.7915 if connection is "wifi" else -0.5818
#     c = 0.2645 if connection is "wifi" else 0.4557

#     coeff = a* (agent.payload)**b + c
#     bits = agent.payload * 1000 * 8
#     P_joule = coeff * bits / 1000 # to convert to mJ

#     return P_joule

def calculate_power_drain(network_latency, connection="wifi"):
    coeff = 0
    if connection == "wifi":
        coeff = hyperparams.MOBILE_WIFI_w
    elif connection == 'lte':
        coeff = hyperparams.MOBILE_4G_w
    else:
        coeff = hyperparams.MOBILE_4G_w

    P_joule = network_latency * coeff

    return P_joule

def critical_mass_threshold(edge_nodes):
    # do sorting and other things
    sorted(edge_nodes, key= lambda x: x.get_machine_load(), reverse=True)
    sorted(edge_nodes, key= lambda x: x.distance_from_next_load_class())
    #s = edge_nodes.sort(key=get_machine_load(len(operator.itemgetter(1))), reverse=True)
    return edge_nodes[0]
                   
def generate_agents(n, merged_raw_ap_data, dim):
    agents = []
    j = 1
    for idx, column in enumerate(merged_raw_ap_data.columns[0:]):
        tot = int(merged_raw_ap_data[column].values[0])
        if tot > 0:
            for _ in range(tot):
                ma_specs = np.random.randint(low=0, high=3, size=1)[0]
                agents.append(MobileAgent(dim, ma_specs, j, column.split(".")[0], idx))
                j += 1 
    # for i in range(n):
    #     ma_specs = np.random.randint(low=0, high=3, size=1)[0]
    #     agents.append(MobileAgent(dim, ma_specs, i))
    return agents

def generate_edge_nodes(EN_RATIO, merged_raw_ap_data, dim):
    enodes = []
    #dt = merged_raw_ap_data.loc[:, (merged_raw_ap_data > 0).all()]
    dt = merged_raw_ap_data
    k = EN_RATIO[0]
    for i in range(len(dt.columns)):
        if k > 0:
            enodes.append(EdgeNode(dim, Tier.One, k, dt.columns[i].split(".")[0], dt.iloc[0, i]))
        else:
            enodes.append(EdgeNode(dim, Tier.AP, k, dt.columns[i].split(".")[0], dt.iloc[0, i]))
        k -= 1
    for i in range(EN_RATIO[1]):
        enodes.append(EdgeNode(dim, Tier.Two, i+EN_RATIO[1]))
    for i in range(EN_RATIO[2]):
        enodes.append(EdgeNode(dim, Tier.Three, i + 2*EN_RATIO[2]))
    return enodes