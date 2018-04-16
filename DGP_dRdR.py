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
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.cross_validation import train_test_split

# Set training size
trainsize = 800
outfile = 'dRdR_200x4.dat'
n_experts = 4
my_njobs = 4

print "Size of the training set is %.3f." % trainsize

#################################################################
# Read in and treat data (mostly) using pandas.                 #
#################################################################

# Read in data (last column is BS)
data_lin = pd.read_csv('/home/ingrid/Documents/Master/ML/Abel_all_crossections/data_40k.dat', sep=' ', skipinitialspace=True)

#Create data frames in pandas
df_lin = pd.DataFrame(data_lin)

# Remove outliers
mask = df_lin["32.dRdR_NLO"] != 0
df_lin = df_lin[mask]

# Lower cut at 10**-16
mask4 = df_lin["32.dRdR_NLO"] > 1e-16
df_lin = df_lin[mask4]

#Check properties of data
print "len(data) = ", len(df_lin)
print "data.shape = ", df_lin.shape

# Define features to train on and target nubers (used below)
# Convert to array for scikit using values
# ravel is used to avoid [[n]], i.e. n x 1 arrays
mean_list = ["39.mcL", "40.mdL", "41.mdR", "42.muL", "43.msL", "44.muR", "45.msR", "46.mcR"]
feature_list = ["38.mGluino", "41.mdR"]

means = df_lin[mean_list].mean(axis=1).values.ravel()

n_features = len(feature_list)
target_list = ["32.dRdR_NLO"]
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

# Define kernel to be used in GP
kernel_matern_3 = C(10, (1e-3, 1000)) * Matern(np.array([1000, 1000, 1000]), (1e3, 1e6), nu=1.5) + WhiteKernel(1, (2e-10,1e-4))

##################

X_train, X_test, y_train, y_test = train_test_split(long_features, target_m2, random_state=42, train_size = trainsize)

N = 20
mq_min = 200
mq_max = 2400
m_rest = 1000

mg_test = np.zeros(N)+500

mq_test = np.linspace(mq_min, mq_max, N)
mean_test = (m_rest*7+mq_test)/8.
mq_test_2 = np.zeros(N)+1000

features_test = np.zeros((N, 3))
features_test_2 = np.zeros((N, 4))

features_test[:,0] = mg_test
features_test[:,1] = mq_test
features_test[:,2] = mean_test

ys = np.zeros(N)+10
# Set name of outfile

# DGP
my_dgp = dgp_parallel(n_experts, outfile, kernel=kernel_matern_3, verbose=False, njobs=my_njobs)
my_dgp.fit_and_predict(long_features, target_m2, trainsize=trainsize, X_train_ = X_train, X_test_ = features_test, y_train_ = y_train, y_test_ = ys)

