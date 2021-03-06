"""
 Created on 19.10.2017

 Reads data files output from harvest_slha.py using pandas and performs ML
 regression with sklearn. Currently uses parallelized Gaussian Processes,
 implemented in the module dgp_parallel.

 @author Ingrid Angelica Vazquez Holm
"""
from dgp_parallel import dgp_parallel
import sys
import numpy as np
import sklearn as sk
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, ConstantKernel as C
import matplotlib.pyplot as plt
import pandas as pd

# Set training size
trainsize =2000

# Choose number of experts
n_experts = 1
my_njobs = 1

print "Size of the training set is %.3f." % trainsize

#################################################################
# Read in and treat data (mostly) using pandas.                 #
#################################################################

# Read in data (last column is BS)
data_lin = pd.read_csv('/home/ingrid/Documents/Master/ML/Abel_all_crossections/data_40k.dat', sep=' ', skipinitialspace=True)

#Create data frames in pandas
df_lin = pd.DataFrame(data_lin)

#Include outliers, but set to -32
#eps = 1e-32
#df_lin = df_lin.replace(0.0, eps)

# Remove outliers
mask = df_lin['17.dLdL_NLO'] != 0
df_lin = df_lin[mask]

# Lower cut at 10**-16
mask4 = df_lin['17.dLdL_NLO'] > 1e-16
df_lin = df_lin[mask4]

#Check properties of data
print "len(data) = ", len(df_lin)
print "data.shape = ", df_lin.shape

# Define features to train on and target nubers (used below)
# Convert to array for scikit using values
# ravel is used to avoid [[n]], i.e. n x 1 arrays
mean_list = ["39.mcL", "40.mdL", "41.mdR", "42.muL", "43.msL", "44.muR", "45.msR", "46.mcR"]
feature_list = ["38.mGluino", "40.mdL"]

means = df_lin[mean_list].mean(axis=1).values.ravel()

n_features = len(feature_list)
target_list = ["17.dLdL_NLO"]
features = df_lin[feature_list].values
target = df_lin[target_list].values.ravel()

long_features = np.zeros(( len(features), 3 ))
long_features[:,0] = features[:,0]
long_features[:,1] = features[:,1]
long_features[:,2] = means[:,]

print "Max target: ", max(np.log10(target))
print "Min target: ", min(np.log10(target))

target_m2 = target/features[:,0]**2
target_mq2 = target/features[:,1]**2
target_fac = target/features[:,0]**2*(features[:,0]**2 +features[:,1]**2)**2

##############################################################
# Distributed Gaussian Processes                             #
##############################################################

# Define kernel to be used in GP
kernel_matern_3 = C(10, (1e-3, 10)) * Matern(np.array([1000, 1000, 1000]), (1e3, 1e6), nu=1.5) + WhiteKernel(1, (2e-10,1e2))
kernel_matern_2 = C(10, (1e-3, 1e3)) * Matern(np.array([1000, 1000]), (1e3, 1e6), nu=1.5) + WhiteKernel(1, (2e-10,1e2))
kernel_rbf_W = C(10, (1e-3, 1e4))*RBF(np.array([1000, 1000]), (1, 1e6)) + WhiteKernel(1, (2e-10,1e2))
kernel_rbf_W_3 = C(10, (1e-3, 1e4))*RBF(np.array([1000, 1000, 1000]), (1, 1e6)) + WhiteKernel(1, (2e-10,1e2))

##################

# Set name of outfile
#outfile = 'bm_dLdL_sigmafac/2000t_nomean_CrbfW_noalpha_.dat'
outfile = 'bm_dLdL_sigmam2/2000t_mean_matern15_noalpha_cut16_smallC.dat'

my_dgp = dgp_parallel(n_experts, outfile, kernel=kernel_matern_3, verbose=False, njobs=my_njobs)#, optimizer=None)
my_dgp.fit_and_predict(long_features, target_m2, trainsize=trainsize)#, alpha=7.544e-07)
#my_dgp.fit_and_predict(features, target_m2, trainsize=trainsize)
