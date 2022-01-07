import enum
import numpy as np
import random
from config import hyperparams

### MOBILE AGENT CLASS ###

class Offloaded(enum.IntEnum):
    Edge = 0
    Cloud = 1
    Local = 2

class MASpecs(enum.IntEnum):
    Low = 0
    Medium = 1
    High = 2 

class MobileAgent(object):
    def __init__(self, area, ma_specs, id=1, ap_name="", ap_id=0):
        self.id = id
        self.ma_specs = MASpecs(ma_specs)
        self.FPS = abs(np.random.choice(hyperparams.FPS))
        self.payload = hyperparams.FRAME_SIZE * self.FPS
        self.max_RTT = 1000 / self.FPS
        
        self.location = self.get_location(area)
        self.required_accuracy = self.get_required_accuracy()
        self.high_mobility = np.random.uniform()
        # self.local_inference_time = np.random.choice(hyperparams.local_inference_dist)
        self.lte_distr = abs(np.random.choice(np.random.normal(5.64, 1, 100)))
        
        self.current_power_drain = self.get_local_power_drain()       
        
        self.network_latency = 0
        self.inference_latency = 0

        self.offload_target = Offloaded.Local
        self.ap_name = ap_name
        self.ap_id = ap_id
        
    # Should be also based on the model required by the agent
    def get_required_accuracy(self):
        return random.randint(60,80)

    # Should be a (lat,lon) couple
    def get_location(self, area):
        return (random.randint(0,area), random.randint(0,area))
    
    def get_estimated_lte_bandwidth(self):
        return self.lte_distr

    # def get_local_power_drain(self):
    #     return int(hyperparams.local_power_drain) * int(self.local_inference_time) * hyperparams.FPS

    def total_latency(self):
        return self.inference_latency + self.network_latency

    def get_local_power_drain(self):
        if self.ma_specs == MASpecs.Low:
            return hyperparams.MOBILE_A_PWR * self.FPS
        elif self.ma_specs == MASpecs.Medium:
            return hyperparams.MOBILE_B_PWR * self.FPS
        else:
            return hyperparams.MOBILE_C_PWR * self.FPS

    def get_mobile_inference_latency(self):
        if self.ma_specs == MASpecs.Low:
            return hyperparams.MOBILE_A_INF
        elif self.ma_specs == MASpecs.Medium:
            return hyperparams.MOBILE_B_INF
        else:
            return hyperparams.MOBILE_C_INF