from entities.en_tier import Tier
import pandas as pd
import numpy as np
import scipy
from scipy import optimize
import pickle


def memoize(func):
    cache = dict()

    def memoized_func(*args):
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result

    return memoized_func

def linear(x, m, b):
    return m*x + b

def inverse_linear(y, m, b):
    return (y - b)/m

def get_queue_time_by_tier(agents=1, tier=Tier.One):
    if tier == Tier.One:
        return linear(agents, *jetson_tx2_queue_time())
    elif tier == Tier.Two:
        return linear(agents, *gtx1060_queue_time())
    else:
        return linear(agents, *rtx2080_queue_time())

def get_max_agents_by_tier(max_RTT=100, tier=Tier.One):
    if tier == Tier.One:
        return inverse_linear(max_RTT, *jetson_tx2_queue_time())
    elif tier == Tier.Two:
        return inverse_linear(max_RTT, *gtx1060_queue_time())
    else:
        return inverse_linear(max_RTT, *rtx2080_queue_time())

def get_compute_time_by_tier(tier=Tier.One):
    if tier == Tier.One:
        return abs(np.random.choice(jetson_tx2_compute_time()))
    elif tier == Tier.Two:
        return abs(np.random.choice(gtx1060_compute_time()))
    else:
        return abs(np.random.choice(rtx2080_compute_time()))

#### TIER 1 ####
################

def jetson_tx2_load_data():
    try:
        with open('data/t1_inference.pkl', 'rb') as f:
            dt = pickle.load(f)
            f.close()
            return dt[0], dt[1]
    except IOError:
        print("File doesn't exist")
        pass

    FOLDER = "../results/jetson_tx2/"
    jetson_data = pd.read_csv(FOLDER + 'mobilenetv2.csv.gz', compression='gzip', header=0, sep=',', quotechar='"', error_bad_lines=False)
    jetson_data = jetson_data[jetson_data['protocol']=='grpc']
    jetson_data = jetson_data[jetson_data['batchsize']==1]
    jetson_data = jetson_data[jetson_data['instance_count']==1]

    jetson_data = jetson_data.assign(request_count_diff=jetson_data.request_count.diff()) \
                    .assign(queue_time_ms=jetson_data.queue_total_time_ns.diff()/(jetson_data.num_clients*10**6)) \
                    .assign(compute_time_ms=jetson_data.compute_total_time_ns.diff()/(jetson_data.num_clients*10**6))

    jetson_data = jetson_data[jetson_data.num_clients == jetson_data.request_count_diff]

    x = np.array([1,2,3,4,5,10,15,20,40])
    grouped = jetson_data.groupby(by=jetson_data.num_clients)
    jetson_tx2_qt_ms = []
    jetson_tx2_qt_ms_mean = []

    for _,s in grouped:
        jetson_tx2_qt_ms.append(s.queue_time_ms)
        jetson_tx2_qt_ms_mean.append(s.queue_time_ms.mean())

    y = np.array(jetson_tx2_qt_ms_mean)
    jetson_tx2_popt, _ = scipy.optimize.curve_fit(linear, x, y, p0=[((75-25)/(44-2)), 0])

    mean = np.mean(jetson_data.compute_time_ms)                  
    variance = np.std(jetson_data.compute_time_ms)

    jetson_tx2_gs = np.random.normal(mean, variance, 1000)

    with open('data/t1_inference.pkl', 'wb') as f:
        print("Saving queue time linear fitting and compute time gaussian for T1 EN")
        pickle.dump([jetson_tx2_popt, jetson_tx2_gs], f)
        f.close()

    return jetson_tx2_popt, jetson_tx2_gs

def jetson_tx2_queue_time():
    jetson_tx2__data_popt, _ = memoized_jetson_tx2_load_data()
    return jetson_tx2__data_popt

def jetson_tx2_compute_time():
    _ , jetson_tx2_gs = memoized_jetson_tx2_load_data()
    return jetson_tx2_gs


#### TIER 2 ####
################

def gtx1060_load_data():
    try:
        with open('data/t2_inference.pkl', 'rb') as f:
            dt = pickle.load(f)
            f.close()
            return dt[0], dt[1]
    except IOError:
        print("File doesn't exist")
        pass

    FOLDER = "../results/gtx1060/"
    gtx1060_data = pd.read_csv(FOLDER + 'mobilenetv2.csv.gz', compression='gzip', header=0, sep=',', quotechar='"', error_bad_lines=False)
    gtx1060_data = gtx1060_data[gtx1060_data['protocol']=='grpc']
    gtx1060_data = gtx1060_data[gtx1060_data['batchsize']==1]
    gtx1060_data = gtx1060_data[gtx1060_data['instance_count']==1]

    gtx1060_data = gtx1060_data.assign(request_count_diff=gtx1060_data.request_count.diff()) \
                    .assign(queue_time_ms=gtx1060_data.queue_total_time_ns.diff()/(gtx1060_data.num_clients*10**6)) \
                    .assign(compute_time_ms=gtx1060_data.compute_total_time_ns.diff()/(gtx1060_data.num_clients*10**6))

    gtx1060_data = gtx1060_data[gtx1060_data.num_clients == gtx1060_data.request_count_diff]

    x = np.array([1,2,3,4,5,10,15,20,40])
    grouped = gtx1060_data.groupby(by=gtx1060_data.num_clients)
    gtx1060_data_qt_ms = []
    gtx1060_data_qt_ms_mean = []

    for _,s in grouped:
        gtx1060_data_qt_ms.append(s.queue_time_ms)
        gtx1060_data_qt_ms_mean.append(s.queue_time_ms.mean())

    y = np.array(gtx1060_data_qt_ms_mean)
    gtx1060_data_popt, _ = scipy.optimize.curve_fit(linear, x, y, p0=[((75-25)/(44-2)), 0])

    mean = np.mean(gtx1060_data.compute_time_ms)                  
    variance = np.std(gtx1060_data.compute_time_ms)

    gtx1060_gs = np.random.normal(mean, variance, 1000)

    with open('data/t2_inference.pkl', 'wb') as f:
        print("Saving queue time linear fitting and compute time gaussian for T2 EN")
        pickle.dump([gtx1060_data_popt, gtx1060_gs], f)
        f.close()

    return gtx1060_data_popt, gtx1060_gs

def gtx1060_queue_time():
    gtx1060_data_popt, _ = memoized_gtx1060_load_data()
    return gtx1060_data_popt

def gtx1060_compute_time():
    _ , gtx1060_gs = memoized_gtx1060_load_data()
    return gtx1060_gs


#### TIER 3 ####
################

def rtx2080_load_data():
    try:
        with open('data/t3_inference.pkl', 'rb') as f:
            dt = pickle.load(f)
            f.close()
            return dt[0], dt[1]
    except IOError:
        print("File doesn't exist")
        pass

    FOLDER = "../results/gpu_server_rtx2080/"
    rtx2080_data = pd.read_csv(FOLDER + 'mobilenetv2_b1_p99_c256.csv', header=0, sep=',', quotechar='"', error_bad_lines=False)
    rtx2080_data_queue_time = rtx2080_data['Server Queue'] / 10**3
    rtx2080_data_compute_time = rtx2080_data['Server Compute'] / 10**3
    rtx2080_data_clients = rtx2080_data['Concurrency']

    rtx2080_x = rtx2080_data_clients.unique()
    rtx2080_y = np.array(rtx2080_data_queue_time)
    rtx2080_data_popt, _ = scipy.optimize.curve_fit(linear, rtx2080_x, rtx2080_y, p0=[((75-25)/(44-2)), 0])

    mean = np.mean(rtx2080_data_compute_time)                  
    variance = np.std(rtx2080_data_compute_time)

    rtx2080_gs = np.random.normal(mean, variance, 1000)

    with open('data/t3_inference.pkl', 'wb') as f:
        print("Saving queue time linear fitting and compute time gaussian for T3 EN")
        pickle.dump([rtx2080_data_popt, rtx2080_gs], f)
        f.close()

    return rtx2080_data_popt, rtx2080_gs

def rtx2080_queue_time():
    rtx2080_data_popt, _ = memoized_rtx2080_load_data()
    return rtx2080_data_popt

def rtx2080_compute_time():
    _ , rtx2080_gs = memoized_rtx2080_load_data()
    return rtx2080_gs


#### MEMOIZE FUNC ####
memoized_jetson_tx2_load_data = memoize(jetson_tx2_load_data)
memoized_rtx2080_load_data = memoize(rtx2080_load_data)
memoized_gtx1060_load_data = memoize(gtx1060_load_data)

#### POPULATING CACHE ####
memoized_jetson_tx2_load_data()
memoized_gtx1060_load_data()
memoized_rtx2080_load_data()