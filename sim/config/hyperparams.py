import time, multiprocessing
import numpy as np
import lib.inference_loader

### LRZ DATA ### 
FOLDER = "data/wlan.lrz-data"
AP_DESCR = "data/full_ap_description.csv"
BUILDING = [
    "LMU, Geb. 0000M, Hauptgebäude/MitteltraktGeschwister-Scholl-Platz 180539 München",
    "LMU, Geb. 0030, Hauptgebäude inkl. Turmgebäude, BibliothekGeschwister-Scholl-Platz 180539 München",
    "LMU, Fachbibliothek PhilologicumLudwigstr. 2580539 München",
    "TUM, Geb. 5406, Chemiegebäude Bau Ch6Lichtenbergstrasse 485748 Garching",
    "HM, Gebäude GLothstr. 3480335 München"
    ]

### LATENCY DATASET ###
SEATTLE_MATRIX_FOLDER = "data/mt/Seattle"
PLANETLAB_MATRIX_FOLDER = "data/mt/PlanetLab"
i11_DATA = "data/ping_exp/data.csv"

### DATA COLLECTION ###
CSV_FOLDER = "data/" + str(time.time())
STORE_RESULTS = False

## FAIR ALLOCATION PARAMETERS ##
alpha = 0.7
beta = 0.3

## MOBILE AGENTS ##
FRAME_SIZE = 33 #KB
FPS = np.random.normal(15, 2, 50)
#FPS = 15 # 300x300x8bit =~ 33KB
#max_RTT = 1000 / FPS # in ms
#local_power_drain = 29.2 # in mJ/ms for inference
#mean_payload = FRAME_SIZE * FPS
#variance_payload = mean_payload * 10**-1

## MobileNetv2 specific parameters from https://dl.acm.org/doi/abs/10.1145/3368305
# Power Consumption in mJoule to TX the raw data (image)
MOBILE_3G = 712
MOBILE_4G = 450
MOBILE_WIFI = 121

# Power coefficient for the network transfer module 
MOBILE_3G_w = 0.8 
MOBILE_4G_w = 2.5
MOBILE_WIFI_w = 1.21

# Power consumption for end-to-end inference on CPU of MobileNetv2 (per frame basically) in mJoule
MOBILE_A_PWR = 182
MOBILE_B_PWR = 318
MOBILE_C_PWR = 268

# Mobile inference time for little CPU for MobileNetv2 in ms
MOBILE_A_INF = 154
MOBILE_B_INF = 116
MOBILE_C_INF = 190

# Heavy-tailed, lognormal distribution for mobile inference time
# This is a lognormal roughly centered around 100
mu = 4.7
sigma = 0.21
local_inference_dist = np.random.lognormal(mu, sigma, 100)

### CLOUD PARAMETERS
cloud_inference = 5 #ms

# ount, bins, ignored = plt.hist(s, 100, density=True, align='mid')

# x = np.linspace(min(bins), max(bins), 10000)
# pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
#        / (x * sigma * np.sqrt(2 * np.pi)))

# plt.plot(x, pdf, linewidth=2, color='r')
# plt.axis('tight')
# plt.show()

## EN NODES ##
T1_RATIO = 2 # The amount of AP which are also T1 EN. 1 T1-EN every T1_RATIO APs
T2_RATI0 = 30 # Number of T2 EN
T3_RATI0 = 3 # Number of T3 EN
# T1_MAX_AGENTS = 4
# T2_MAX_AGENTS = 16
# T3_MAX_AGENTS = 64

## EXPERIMENT PARAMETERS ##
dataset_rtt = "Seattle" # Can be Seattle, PlanetLab, i11, or None to use semi-random latency matrix.
save_plots = True
minimum_agents_threshold = 30
greedy = False
tolerance = 0.9

# A global search for the best offloading candidate can be very time consuming with a lot of APs.
# By setting local_search_scope to a value > 0, we limit the search to the best k EN sorted by RTT. 
local_search_scope = 0

parallel_solvers = multiprocessing.cpu_count() - 2 # Keep some cores free!