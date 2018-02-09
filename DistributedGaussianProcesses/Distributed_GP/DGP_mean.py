"""
 Created on 19.10.2017

 Reads data files output from harvest_slha.py using pandas and performs ML
 regression with sklearn. Currently uses parallelized Gaussian Processes,
 implemented in the module pdg_parallel.

 To run the script mpi must be used: 

python DGP_phys.py trainsize outfile kernel experts nodes

 @author Ingrid Angelica Vazquez Holm
"""

from dgp_parallel import dgp_parallel
import sys
import numpy as np
import sklearn as sk
from sklearn.gaussian_process.kernels import RationalQuadratic, RBF, WhiteKernel, ExpSineSquared, DotProduct, Matern, ConstantKernel as C

import matplotlib.pyplot as plt
import pandas as pd

# Set training size
if len(sys.argv) >= 2:
    trainsize = float(sys.argv[1])
    if trainsize >= 1:
        trainsize = int(trainsize)
else:
    trainsize = 0.001

# Set name of outfile
if len(sys.argv) >= 3:
    outfile = sys.argv[2]
else:
    print "Error! No outfile was provided."
    sys.exit(0)

# Set kernel
if len(sys.argv) >= 4:
    my_kernel = str(sys.argv[3])
else:
    print "Error! No kernel was provided."
    sys.exit(1)

# Choose number of experts
if len(sys.argv) >= 5:
    n_experts = int(sys.argv[4])
else:
    n_experts = 4

if len(sys.argv) >= 6:
    my_njobs = int(sys.argv[5])
else:
    my_njobs = 1

print "Size of the training set is %.3f of the total dataset." % trainsize

#################################################################
# Read in and treat data (mostly) using pandas.                 #
#################################################################

# Read in data (last column is BS)
#data_lin = pd.read_csv('/home/ingrid/Documents/Master/ML/Abel_long/data_40k.dat', sep=' ', skipinitialspace=True)

# Lin
#data_lin = pd.read_csv('/home/ingrid/Documents/Master/ML/Abel_long/data_20k_lin_1.dat', sep=' ', skipinitialspace=True)

# Log
data_lin = pd.read_csv('/home/ingrid/Documents/Master/ML/Abel_long/data_20k_log_8.dat', sep=' ', skipinitialspace=True)

#Create data frames in pandas
df_lin = pd.DataFrame(data_lin)

#Drop BS column
df_lin = df_lin.drop('Unnamed: 11', axis=1)
mask = df_lin['2.qq_NLO'] != 0
df_lin = df_lin[mask]

###############################
# Choose region               #
###############################

# Lower limit
mask4 = df_lin['2.qq_NLO'] > 1e-16
df_lin = df_lin[mask4]

#Check properties of data
print "len(data) = ", len(df_lin)
print "data.shape = ", df_lin.shape

# Define features to train on and target nubers (used below)
# Convert to array for scikit using values
# ravel is used to avoid [[n]], i.e. n x 1 arrays
mean_list = [ "4.mcL", "5.mdL", "6.mdR", "7.muL", "8.msL", "9.muR", "10.msR", "11.mcR"]
feature_list = ["3.mGluino", "4.mcL"]

means = df_lin[mean_list].mean(axis=1).values.ravel()

n_features = len(feature_list)
target_list = ["2.qq_NLO"]
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
kernel_matern1 = C(10, (1e-3, 100)) * Matern(np.array([10, 10, 10]), (1e-4, 1e6), nu=1.0)
kernel_matern2 = C(10, (1e-3, 1e3)) * Matern(np.array([10, 10]), (1e-4, 1e6), nu=1.0)
kernel_fixed = C(10.8**2) * Matern(length_scale=[9.15e+04, 4.78e+03, 1e+06], nu=1) 

kernel_linmin16_logmin9 = C(10**2) * Matern(length_scale=[2.25e+05, 5.05])

# mean_2 : 11.2**2 * Matern(length_scale=[1.64e+05, 4.29e+03, 3.65e+05], nu=1)
# mean_3 : 10.8**2 * Matern(length_scale=[9.15e+04, 4.78e+03, 1e+06], nu=1)
# 20k data max6 : 11.8**2 * Matern(length_scale=[1.66e+05, 1.15e+03, 7.36e+05], nu=1)

##################

# Set the wanted kernel
if my_kernel == 'kernel_matern1': my_kernel = kernel_matern1
elif my_kernel == 'kernel_matern2' : my_kernel = kernel_matern2
elif my_kernel == 'kernel_fixed' : my_kernel = kernel_fixed
else:
    print "Error! Not a valid kernel."
    sys.exit(2)

my_dgp = dgp_parallel(n_experts, outfile, kernel=my_kernel, verbose=False, njobs=my_njobs)#, optimizer=None)
my_dgp.fit_and_predict(long_features, target_m2, trainsize=trainsize, alpha=7.544e-07)

"""
min 1e-20 on lin20k 

Kernel parameters:  2.98**2 * Matern(length_scale=[6.69e+04, 6.76e+03, 1e+05], nu=1)
Delta Time GP fit 110.838976145
Kernel parameters:  3.36**2 * Matern(length_scale=[6.57e+04, 7.42e+03, 1e+05], nu=1)
Time for filling in mu and sigma  53.9891498089
Time for filling in mu and sigma  55.1240680218
/home/ingrid/.local/lib/python2.7/site-packages/sklearn/gaussian_process/gpr.py:457: UserWarning: fmin_l_bfgs_b terminated abnormally with the  state: {'warnflag': 2, 'task': 'ABNORMAL_TERMINATION_IN_LNSRCH', 'grad': array([  -0.15020965,    4.30617911,    0.94361196, -116.63575518]), 'nit': 13, 'funcalls': 97}
  " state: %s" % convergence_dict)
Delta Time GP fit 279.57449317
Kernel parameters:  3.03**2 * Matern(length_scale=[6.11e+04, 7.12e+03, 1e+05], nu=1)
Delta Time GP fit 280.648066998
Kernel parameters:  3**2 * Matern(length_scale=[6.29e+04, 7.09e+03, 1e+05], nu=1)
"""
