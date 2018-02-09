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
from sklearn.gaussian_process.kernels import RationalQuadratic, RBF, WhiteKernel, ExpSineSquared, DotProduct, Matern, ConstantKernel as C, Exponentiation
from sklearn.gaussian_process.kernels import PairwiseKernel
from sklearn.metrics.pairwise import chi2_kernel

#Valid metrics for PairwiseKernel:
#['rbf', 'sigmoid', 'polynomial', 'poly', 'linear', 'cosine']

from sklearn.metrics.pairwise import cosine_similarity, linear_kernel, sigmoid_kernel

import time
import pandas as pd

#Small number
eps = 1E-32 # Used to regularize points with zero cross section

# Set training size
if len(sys.argv) >= 2:
    trainsize = float(sys.argv[1])
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

t0 = time.time()
print "Starting process at time", t0
print "Size of the training set is %.3f of the total dataset." % trainsize

#################################################################
# Read in and treat data (mostly) using pandas.                 #
#################################################################

# Read in data (last column is BS)
data_lin = pd.read_csv('/home/ingrid/Documents/Master/ML/Abel_test_params/log_7.dat', sep=' ', skipinitialspace=True)

print "Read in data at time ", time.time() - t0

#Create data frames in pandas
df_lin = pd.DataFrame(data_lin)

#Drop BS column
df_lin = df_lin.drop('Unnamed: 5', axis=1)
mask = df_lin['2.qq_NLO'] != 0
df_lin = df_lin[mask]

#Drop cross sections under -2
mask2 = df_lin['2.qq_NLO'] >= 1e-2
df_lin = df_lin[mask2]

#Find zero cross sections and replace with small number
#df_lin = df_lin.replace(0.0, eps)

print "Massaged data at time ", time.time() - t0

#Check properties of data
print "len(data) = ", len(df_lin)
print "data.shape = ", df_lin.shape

# Define features to train on and target nubers (used below)
# Convert to array for scikit using values
# ravel is used to avoid [[n]], i.e. n x 1 arrays
feature_list = ["4.mGluino", "5.mcL"]
n_features = len(feature_list)
target_list = ["2.qq_NLO"]
features = df_lin[feature_list].values
target = df_lin[target_list].values.ravel()

print "Len: ", len(target)
print "Max: ", max(target)
print "Min: ", min(target)

# Alternative features,[0]=m_ and [1]=beta
# Recall that masses are in GeV, so s = 8**2 TeV = 64e+06 GeV
s = 64e+06 # 8TeV
features_alt = np.zeros((len(features), 2))
features_alt_2 = np.zeros((len(features), 2))
for i in range(len(features)):
    features_alt[i][0] = features[i][0]**2 - features[i][1]**2
    features_alt_2[i][0] = features[i][0]
    if 4*features[i][1]**2 <= s:
        features_alt[i][1] = np.sqrt(1 - 4*features[i][1]**2/s)
        features_alt_2[i][1] = np.sqrt(1 - 4*features[i][1]**2/s)
    else:
        features_alt[i][1] = 0

m_ = features[:,0]**2 - features[:,1]**2
features_msq = features[:,1]
features_sq = np.zeros((len(features), 2))
for i in range(len(features)):
    features_sq[i][0] = features[i][0]**2
    features_sq[i][1] = features[i][1]**2


betaq = np.sqrt(1-4*features[:,1]**2/s)
L1 = np.log((s+2*m_- s*betaq)/(s+2*m_+s*betaq))
target_beta = target/betaq
target_L1 = np.zeros((len(target)))
for i in range(len(target)):
    if L1[i] != 0:
        target_L1[i] = target[i]/abs(L1[i])

target_mg = target/features[:,0]
target_m2 = target/features[:,0]**2
target_mg2_beta = target/(features[:,0]**2*betaq)
target_m_ = target/m_
target_s = target*s
target_mq = target*features[:,1]**2
target_mg2_pos = target*(features[:,0]**2)
target_mr = target*(features[:,1]**4/features[:,0]**2)
target_mq4 = target*features[:,1]**4

##############################################################
# Distributed Gaussian Processes                             #
##############################################################

# Define kernel to be used in GP
kernel6 = C(1.0, (1e-2, 1e8)) * RBF(10, (1e-4, 1e4)) + C(0.001, (1e-6,1e5))

kernel_matern1 = C(10, (1e-3, 1e3)) * Matern(np.array([10, 1.0]), (1e-4, 1e6), nu=1) + C(0.001, (1e-6,1e5))
kernel_matern15 = C(10, (1e-3, 1e3)) * Matern(np.array([10, 1.0]), (1e-4, 1e6), nu=1.5) + C(0.001, (1e-6,1e5))
kernel_matern2 = C(10, (1e-3, 1e3)) * Matern(np.array([10, 1.0]), (1e-4, 1e6), nu=2) + C(0.001, (1e-6,1e5))
kernel_matern25 = C(10, (1e-3, 1e3)) * Matern(np.array([10, 1.0]), (1e-4, 1e6), nu=2.5) + C(0.001, (1e-6,1e5))

kernel_matern_rq =  C(10, (1e-3, 1e3)) * Matern(np.array([10, 1.0]), (1e-4, 1e6), nu=1) + C(1, (1e-3, 10)) * RationalQuadratic(0.002, 0.1,(1e-7, 1.0), (1e-7, 1.0))

kernel_matern_rbf = C(10, (1e-3, 1e3)) * Matern(np.array([10, 1.0]), (1e-4, 1e6), nu=1) + C(2, (1e-3, 1e2))* RBF(1, (1e-4, 0.1)) +  C(2, (1e-3, 1e2))


kernel_rationalquad = C(10, (1e-3, 1e3)) * RationalQuadratic(10.0, 1.0, (1e-3, 1e5), (1e-3, 1e3)) + C(1.0, (1e-4, 1e5))

kernel_sum =  C(1.0, (1e-2, 1e3)) * RBF(10, (1e-4, 1e4)) + C(1.0, (1e-2, 1e3)) * RBF(10, (1e-4, 1e3)) + C(1.0, (1e-2, 1e8)) * RBF(10, (1e-4, 1e4)) + C(1.0, (1e-2, 1e3)) * RBF(10, (1e-4, 1e3)) + C(0.001, (1e-6,1e5))

kernel_RQ = C(10, (1e-3, 1e3)) * RBF(np.array([1e5, 1.0]), (1e-5, 1e11)) * RationalQuadratic(10.0, 1.0, (1e-3, 1e5), (1e-3, 1e3)) + C(1.0, (1e-4, 1e5)) #* ExpSineSquared(1.0, 1.0 , (1e-3, 1e3), (1e-3, 1e3))

kernel_ell = C(1.0, (1e-2, 1e8)) * RBF(np.array([10, 1.0]), (1e-4, 1e4)) + C(0.001, (1e-6,1e5))
kernel_ell_wk = C(1.0, (1e-2, 1e8)) * RBF(np.array([10, 1.0]), (1e-4, 1e4)) + WhiteKernel(0.001, (1e-8,10))

kernel_phys = C(10, (1e-3, 1e3)) * Matern(np.array([10, 1.0]), (1e-4, 1e6), nu=1.5) + C(1.0, (1e-2, 1e8)) * RBF(np.array([10, 1.0]), (1e-4, 1e4))   + C(0.001, (1e-6,1e5))

kernel_decades =  C(20, (1, 1e2)) * RBF(np.array([700, 1000]), (1e2, 1e4)) + C(1.0, (1e-1, 10)) * RBF(np.array([200, 10]), (1e-3, 300)) + C(0.01, (1e-3,1e3))

kernel_mat_rbf = C(10, (1e-3, 1e3)) * Matern(np.array([10, 1.0]), (1e-4, 1e6), nu=1) * RBF(10, (1e-4, 1e4)) + C(1.0, (1e-2, 1e3)) 

kernel_mat_rbfell = C(10, (1e-3, 1e3)) * Matern(np.array([10, 1.0]), (1e-4, 1e6), nu=1.5) * RBF(np.array([10, 1.0]), (1e-4, 1e4)) + C(1.0, (1e-2, 1e3))

kernel_dot_rbf = C(20, (1, 1e2)) * RBF(np.array([700, 1000]), (1e2, 1e4)) * DotProduct(0.5, (1e-3, 1e2)) + C(1.0, (1e-1, 10))
kernel_dot = C(20, (1, 1e2)) * DotProduct(0.5, (1e-3, 1e2)) + C(1.0, (1e-1, 10))
kernel_exp =  C(20, (1, 1e2)) * Exponentiation( RBF(np.array([700, 1000]), (1e2, 1e4)), 3)
pairwise = PairwiseKernel(metric='polynomial', gamma=1.0, gamma_bounds=(1e-7, 1e5)) #, pairwise_kernels_kwargs={'coef0':0})

print "Kernel theta: ", pairwise.theta

# Set the wanted kernel
if my_kernel == 'kernel4':    my_kernel = kernel4
elif my_kernel == 'kernel5':    my_kernel = kernel5
elif my_kernel == 'kernel6':    my_kernel = kernel6
elif my_kernel == 'kernel7':    my_kernel = kernel7
elif my_kernel == 'kernel_ell': my_kernel = kernel_ell
elif my_kernel == 'kernel_RQ': my_kernel = kernel_RQ
elif my_kernel == 'kernel_matern1': my_kernel = kernel_matern1
elif my_kernel == 'kernel_matern15': my_kernel = kernel_matern15
elif my_kernel == 'kernel_matern2': my_kernel = kernel_matern2
elif my_kernel == 'kernel_matern25': my_kernel = kernel_matern25
elif my_kernel == 'kernel_rationalquad': my_kernel = kernel_rationalquad
elif my_kernel == 'kernel_sum': my_kernel = kernel_sum
elif my_kernel == 'kernel_phys': my_kernel = kernel_phys
elif my_kernel == 'kernel_decades': my_kernel = kernel_decades
elif my_kernel == 'kernel_mat_rbf' : my_kernel = kernel_mat_rbf
elif my_kernel == 'kernel_mat_rbfell' : my_kernel = kernel_mat_rbfell
elif my_kernel == 'kernel_matern_rq' : my_kernel = kernel_matern_rq
elif my_kernel == 'kernel_matern_rbf' : my_kernel = kernel_matern_rbf
elif my_kernel == 'kernel_dot_rbf' : my_kernel = kernel_dot_rbf
elif my_kernel == 'kernel_dot' : my_kernel = kernel_dot
elif my_kernel == 'kernel_exp' : my_kernel = kernel_exp
elif my_kernel == 'pairwise' : my_kernel = pairwise
elif my_kernel == 'kernel_ell_wk' : my_kernel = kernel_ell_wk
else:
    print "Error! Not a valid kernel."
    sys.exit(2)


print "Starting gaussian process analysis...\n"
print "The time before starting here is ", time.time() - t0
 
my_dgp = dgp_parallel(n_experts, outfile, kernel=my_kernel, verbose=False, njobs=my_njobs)
my_dgp.fit_and_predict(features, target_mr, trainsize=trainsize, alpha=0.002)
