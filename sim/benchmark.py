import numpy as np
import random as rand

# Add code to test the solver with multiple parameters and store the results plus the plots

#####################################
# MOBILE AGENT PARAMETERS           #
#####################################

max_RTT = np.arange(100, 300, 25)
payload = np.arange(0.1, 0.5, 0.1)
local_power_drain = 29.2
local_inference_time = np.arange(50, 200, 25)
requests_per_second = 1
payload = abs(np.random.choice(np.random.normal(500, 100, 100))/1000)

#####################################
# EDGE NODES PARAMETERS             #
#####################################

# Example device: Razer laptop
max_servable_agents_small = np.arange(2, 16, 2)
execution_time_small = np.arange(70, 120, 10)
wifi_bw_small = np.random.choice(np.random.normal(300, 10, 100))
parallelism_factor_small = 2

# Example device: Micro-server
max_servable_agents_medium = np.arange(16, 32, 2)
execution_time_medium = np.arange(40, 70, 10)
wifi_bw_medium = np.random.choice(np.random.normal(1000, 50, 100))
parallelism_factor_medium = 4

# Example device: Cloud instance
max_servable_agents_high = np.arange(64, 128, 16)
execution_time_medium = np.arange(10, 20, 2)
wifi_bw_high = np.random.choice(np.random.normal(5000, 50, 100))
parallelism_factor_medium = 10

#####################################
# CLOUD PARAMETERS                  #
#####################################

cloud_exec_time = 10


def run_benchmark():
    # Make assumptions -> manually set some parameters and fix them
    # remove request per second parameter and just use fixed payload
    # modify the others and call the solver
    return None