
import os, pickle
import pandas as pd
import numpy as np
import random as rand
import seaborn as sns
import matplotlib.pyplot as plt
import config.hyperparams as hp
from sklearn import mixture
from tqdm import tqdm

# Requires data from this repo: https://github.com/uofa-rzhu3/NetLatency-Data.git
# Data from both datasets exhibit very high lateny
# Plotting network graphs: https://plot.ly/python/network-graphs/
# 4G RTT measurement http://dl.ifip.org/db/conf/im/im2019-ws1-annet/191661.pdf

# Load Seattle data
def load_seattle_data(slice_id=None):
    filen = "/SeattleData_" + str(rand.randint(1, 688)) if not slice_id else "/SeattleData_" + str(slice_id)
    dt = pd.read_csv(hp.SEATTLE_MATRIX_FOLDER + filen, sep="\s+", header = None)
    dt.clip(0, 0.25, inplace=True)
    return dt

# LoadPlanetLab data
def load_planetlab_data(slice_id=None):
    filen = "/PlanetLabData_" + str(rand.randint(1, 18)) if not slice_id else "/PlanetLabData_" + str(slice_id)
    dt = pd.read_csv(hp.PLANETLAB_MATRIX_FOLDER + filen, sep="\s+", header = None)
    dt.clip(0, 250, inplace=True)
    return dt

def i11_data():
    dt = pd.read_csv(hp.i11_DATA, sep="\s+", header = None)
    return dt

def generate_data(fitted_gaussian_distr, n=99, m=99):
    # Generates a matrix of size nxm populated with rtt values compatible with the selected datasource
    rtt, _ = fitted_gaussian_distr.sample(n*m)
    rtt_matrix = abs(rtt.reshape(n,m))

    return rtt_matrix


# https://www.python-course.eu/expectation_maximization_and_gaussian_mixture_models.php
def analyze_data_old(source="Seattle", threshold=250):
    from scipy.stats import norm

    frames = []
    print("Loading dataset ...")
    if source is "Seattle":
        for fl in tqdm(os.listdir(hp.SEATTLE_MATRIX_FOLDER), total=688):
            frames.append(pd.read_csv(hp.SEATTLE_MATRIX_FOLDER + "/" + fl, sep="\s+", header = None))
    else:
        for fl in tqdm(os.listdir(hp.PLANETLAB_MATRIX_FOLDER), total=18):
            frames.append(pd.read_csv(hp.PLANETLAB_MATRIX_FOLDER + "/" + fl, sep="\s+", header = None))
    
    print("Stacking frames ...")
    res = pd.concat(frames)
    # Multiply by 1000 to transform values to milliseconds
    res_n = res.to_numpy().flatten() * 1000
    # Remove values above a threshold
    res_n = res_n[res_n <= threshold]

    _, ax = plt.subplots(3, 1, figsize=(12, 6))
    sns.distplot(res_n, bins=10, ax=ax[0])

    # Create the array r with dimensionality nxK
    r = np.zeros((len(res_n), 3))  
    print('Dimensionality','=',np.shape(r))

    # Instantiate the random gaussians
    gauss_1 = norm(loc=40,scale=20) 
    gauss_2 = norm(loc=150,scale=30)
    gauss_3 = norm(loc=400,scale=70)

    # Instantiate the random pi_c
    pi = np.array([1/3,1/3,1/3]) # We expect to have three clusters 

    #Probability for each datapoint x_i to belong to gaussian g
    print("Calculating probability for each datapoint of belonging to one of tha Gaussians") 
    for c,g,p in zip(range(3),[gauss_1,gauss_2,gauss_3], pi):
        r[:,c] = p*g.pdf(res_n) # Write the probability that x belongs to gaussian c in column c. 
    # Therewith we get a 60x3 array filled with the probability that each x_i belongs to one of the gaussians

    # Normalize the probabilities such that each row of r sums to 1
    print("Normalizing probabilities") 
    a = np.sum(pi)
    b = np.sum(r,axis=1)
    for i in tqdm(range(len(r))):
        # r[i] = Div(r[i], (np.sum(pi)*np.sum(r,axis=1)[i]))
        r[i] = r[i]/(a*b[i])

    # Plot the data and suppressing warning for scatter plot
    from matplotlib.axes._axes import _log as matplotlib_axes_logger
    matplotlib_axes_logger.setLevel('ERROR')

    for i in range(len(r)):
        ax[2].scatter(res_n[i],0,c=np.array([r[i][0],r[i][1],r[i][2]]),s=100) # We have defined the first column as red, the second as
                                                                    # green and the third as blue
    for g,c in zip([gauss_1.pdf(np.linspace(0,1000)),gauss_2.pdf(np.linspace(0,1000)),gauss_3.pdf(np.linspace(-15,15))],['r','g','b']):
        ax[2].plot(np.linspace(0,1000),g,c=c,zorder=0)

    plt.show()
    
    # Tenative fitting
    d1 = np.random.normal(40, 20, 1000)
    d2 = np.random.normal(150, 30, 1000)
    d3 = np.random.normal(400, 70, 1000)
    sns.distplot(d1, hist=False, rug=True, ax=ax[1])
    sns.distplot(d2, hist=False, rug=True, ax=ax[1])
    sns.distplot(d3, hist=False, rug=True, ax=ax[1])


def analyze_data(source="Seattle", drange=1000, threshold=250, k=3, quiet=True):
    frames = []
    try:
        with open('data/gmm_{}.pkl'.format(source), 'rb') as f:
            clf = pickle.load(f)
            f.close()
            return clf[0]
    except IOError:
        print("File doesn't exist")
        pass

    print("Loading dataset ...")
    if source is "Seattle":
        for fl in tqdm(os.listdir(hp.SEATTLE_MATRIX_FOLDER), total=688):
            frames.append(pd.read_csv(hp.SEATTLE_MATRIX_FOLDER + "/" + fl, sep="\s+", header = None))
    elif source is "PlanetLab":
        for fl in tqdm(os.listdir(hp.PLANETLAB_MATRIX_FOLDER), total=18):
            frames.append(pd.read_csv(hp.PLANETLAB_MATRIX_FOLDER + "/" + fl, sep="\s+", header = None))
    else:
         frames.append(pd.read_csv(hp.i11_DATA, header = None))
    
    print("Stacking frames ...")
    rtt = pd.concat(frames)
    rtt = rtt.to_numpy().flatten()

    if source is "Seattle":
        # Multiply by 1000 to transform values to milliseconds
        rtt = rtt * 1000
    
    # Remove values above a threshold
    rtt = rtt[rtt <= threshold]
    rtt = rtt.reshape(-1, 1)

    print("GMM Fitting ...")
    clf = mixture.GaussianMixture(n_components=k, covariance_type='full')
    clf.fit(rtt)

    variance = [np.sqrt(np.trace(clf.covariances_[i])/k) for i in range(0,k)]
    mean = clf.means_

    # Saving the objects:
    with open('data/gmm_{}.pkl'.format(source), 'wb') as f:
        print("Saving GMM fitting results")
        pickle.dump([clf], f)
        f.close()

    if not quiet:
        print("Plotting ...")
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        #sns.distplot(rtt, bins=10, ax=ax[0])

        for i in range(0,k):
            t = np.random.normal(mean[i], variance[i], drange)
            sns.distplot(t, hist=False, rug=True, ax=ax, label="Group {}".format(i))
        
        ax.set_xlabel("[ms]")
        fig.savefig("../plots/latency_distr_{}.pdf".format(source))
        
    return clf

def plot_matrix():
    print("Plotting ...")
    fig, ax = plt.subplots(2, 2, figsize=(12, 6))

    rtt_matrix_seattle = load_seattle_data()
    rtt_matrix_planetlab = load_planetlab_data()

    rtt_matrix_planetlab.clip(0, 250, inplace=True)
    rtt_matrix_seattle.clip(0, 0.25, inplace=True)

    #sns.distplot(rtt_matrix_seattle, bins=10, ax=ax[0,0])
    sns.heatmap(rtt_matrix_seattle, ax=ax[0,1])

    #sns.distplot(rtt_matrix_planetlab, bins=10, ax=ax[1,0])
    sns.heatmap(rtt_matrix_planetlab, ax=ax[1,1])

    fig.tight_layout(w_pad=1.5)