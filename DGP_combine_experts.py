"""
 Created on 19.10.2017

 Reads data files output from harvest_slha.py using pandas and performs ML
 regression with sklearn. Currently uses parallelized Gaussian Processes,
 implemented in the module pdg_parallel.

 To run the script mpi must be used: 

python DGP_phys.py trainsize outfile kernel experts nodes

 @author Ingrid Angelica Vazquez Holm
"""

#from dgp_parallel import dgp_parallel
import sys
import numpy as np
import sklearn as sk
from sklearn.gaussian_process.kernels import RationalQuadratic, RBF, WhiteKernel, ExpSineSquared, DotProduct, Matern, ConstantKernel as C
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import pandas as pd

#################################################################
# Read in and treat data (mostly) using pandas.                 #
#################################################################

target_list = "23.uLuL_NLO"
feature_list = ["38.mGluino", "42.muL"]
testsize = 1000

# Read in data (last column is BS)
data_lin = pd.read_csv('/home/ingrid/Documents/Master/Programs/abel_data_290k.dat', sep=' ', skipinitialspace=True)

#Create data frames in pandas
df_lin = pd.DataFrame(data_lin)

mask = df_lin[target_list] > 1e-16
df_lin = df_lin[mask]

###############################
# Choose region               #
###############################

#Check properties of data
print "len(data) = ", len(df_lin)
print "data.shape = ", df_lin.shape

# Define features to train on and target nubers (used below)
# Convert to array for scikit using values
# ravel is used to avoid [[n]], i.e. n x 1 arrays
mean_list = ["39.mcL", "40.mdL", "41.mdR", "42.muL", "43.msL", "44.muR", "45.msR", "46.mcR"]
means = df_lin[mean_list].mean(axis=1).values.ravel()

n_features = len(feature_list)
features = df_lin[feature_list].values
target = df_lin[target_list].values.ravel()

long_features = np.zeros(( len(features), 3 ))
long_features[:,0] = features[:,0]
long_features[:,1] = features[:,1]
long_features[:,2] = means[:,]

print "Max target: ", max(np.log10(target))
print "Min target: ", min(np.log10(target))

target_m2 = target/features[:,0]**2

##############################################################
# Distributed Gaussian Processes                             #
##############################################################

X_train, X_test, y_train, y_test = train_test_split(long_features, target, random_state=42, test_size=testsize)

from dgp_predict import dgp_predict
models = ['uLuL_2000_1', 'uLuL_2000_2', 'uLuL_2000_3', 'uLuL_2000_4', 'uLuL_2000_5', 'uLuL_2000_6', 'uLuL_2000_7', 'uLuL_2000_8', 'uLuL_2000_9', 'uLuL_2000_10']

my_dgp = dgp_predict(models)
mu, sigma = my_dgp.predict(X_test)

mu += 2*np.log10(X_test[:,0])

errors = (y_test - 10**mu)/y_test

d = {'Errors' : errors, 'Mus' : mu, 'Sigmas' : sigma, 'Y_test' : y_test, 'Mg' : X_test[:,0], 'Mq' : X_test[:,1]}
df = pd.DataFrame(d)
df.to_csv('uLuL_2000x10_mssm.dat', sep=' ')

#plt.scatter(X_test[:,1], np.log10(y_test), color='blue', alpha=0.3)
#plt.scatter(X_test[:,1], mu, color='red', alpha=0.3)
#plt.show()
