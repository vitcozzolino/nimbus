import mysql.connector
import numpy as np
import seaborn as sns
import os, pickle

def init_connection(host="localhost", db="ripe_measurements"):
    print("Initializing connectiong with DB")
    mydb = mysql.connector.connect(
    host=host,
    user="root",
    passwd="mysqlgpuserver",
    database=db
    )
    return mydb

def get_cloud_rtt(datacenter_region="Europe", probe_region='DE', drange=1000):
    try:
        with open('data/cloud_gm_{}_{}.pkl'.format(datacenter_region, probe_region), 'rb') as f:
            #print("Loading pre-calculated model")
            gs = pickle.load(f)
            f.close()
            return gs[0]
    except IOError:
        print("Gaussian model for datacenter {} and probe region {} doesn't exist".format(datacenter_region, probe_region))
        pass
    
    mydb = init_connection()
    mycursor = mydb.cursor()
    query = "SELECT ripe_measurements.pingMeasurements.ping1 FROM ripe_measurements.endpoints" \
            " RIGHT JOIN ripe_measurements.pingMeasurements on" \
            " ripe_measurements.endpoints.epAddress = ripe_measurements.pingMeasurements.epAddress" \
            " RIGHT JOIN ripe_measurements.probes on" \
            " ripe_measurements.probes.probeId = ripe_measurements.pingMeasurements.probeId" \
            " where epContinent='{}' and probeCountry='{}'".format(datacenter_region, probe_region)
    print("Executing query now .. ")
    mycursor.execute(query)
    result = mycursor.fetchall() 

    #TODO: Groupby cloud provider?
    result = list(map(lambda x: x[0] if x[0] is not None else 0, result))

    mean = np.mean(result)                  
    variance = np.std(result)

    gs = np.random.normal(mean, variance, drange)

    # Saving the objects:
    with open('data/cloud_gm_{}_{}.pkl'.format(datacenter_region, probe_region), 'wb') as f:
        print("Saving gaussian model")
        pickle.dump([gs], f)
        f.close()

    sns.distplot(gs, hist=False, rug=True)

    return gs