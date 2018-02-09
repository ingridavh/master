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
from sklearn.cross_validation import train_test_split
from sklearn.gaussian_process.kernels import RationalQuadratic, RBF, WhiteKernel, ExpSineSquared, DotProduct, Matern, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

#################################################################
# Read in and treat data (mostly) using pandas.                 #
#################################################################

# Read in data (last column is BS)
data_lin = pd.read_csv('/home/ingrid/Documents/Master/ML/Abel_20k_physical/data_20k_phys.dat', sep=' ', skipinitialspace=True)

#Create data frames in pandas
df_lin = pd.DataFrame(data_lin)
mask = df_lin['2.qq_NLO'] != 0
df_lin = df_lin[mask]

#Drop BS column
df_lin = df_lin.drop('Unnamed: 4', axis=1)

#Check properties of data
print "len(data) = ", len(df_lin)
print "data.shape = ", df_lin.shape

# Define features to train on and target nubers (used below)
# Convert to array for scikit using values
# ravel is used to avoid [[n]], i.e. n x 1 arrays
feature_list = ["3.mGluino", "4.mcL"]
n_features = len(feature_list)
target_list = ["2.qq_NLO"]
features = df_lin[feature_list].values
target = df_lin[target_list].values.ravel()

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
betaq = np.sqrt(1-4*features[:,1]**2/s)
L1 = np.log((s+2*m_- s*betaq)/(s+2*m_+s*betaq))
target_beta = target/betaq
target_L1 = np.zeros((len(target)))
for i in range(len(target)):
    if L1[i] != 0:
        target_L1[i] = target[i]/abs(L1[i])

target_mg = target/features[:,0]
target_mg2 = target/features[:,0]**2
target_mg2_beta = target/(features[:,0]**2*betaq)
target_m_ = target/m_
target_s = target*s
target_mq = target*(features[:,1]**2)
target_mg2_pos = target*(features[:,0]**2)
target_mr = target*(features[:,1]**4/features[:,0]**2)

features_short = features[:500]
target_m2_short = target_mg2[:500]

##############################################################
# Distributed Gaussian Processes                             #
##############################################################

# Define kernel to be used in GP
kernel6 = C(1.0, (1e-2, 1e8)) * RBF(10, (1e-4, 1e4)) + C(0.001, (1e-6,1e5))
kernel_matern = C(10, (1e-3, 1e3)) * Matern(np.array([10, 1.0]), (1e-4, 1e6), nu=1) + C(0.001, (1e-6,1e5))
matern = 25.0**2*Matern(10, (1e-4, 1e6), nu=1)+C(0.1, (1e-4, 1e+2))
matern_ell = 25.0**2*Matern(np.array([100, 100]), (1e2, 1e6), nu=1) 
kernel_ell = C(1.0, (1e-2, 1e8)) * RBF(np.array([100, 100]), (1e-1, 1e6))# + C(0.001, (1e-6,1e5))
rbf = C(1.0, (1e-1,1e+03)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e4))+ WhiteKernel(0.1, (1e-4, 1e+2))

matern_rbf = 25.0**2 * Matern(100, (1, 1e6), nu=1) * RBF(10, (1e-1, 1e4))

X = features_short
y = np.log10(target_m2_short)

print min(y), max(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state=42)

gp = GaussianProcessRegressor(kernel=matern_ell, random_state=42, alpha=7.544e-07).fit(X_train, y_train)
#gp.predict(features, target, trainsize=trainsize, alpha=0.01)
print gp.get_params()
print gp.kernel_
#N = 10

# Plot LML landscape
theta0 = np.logspace(0, 5, 49) 
theta1 = np.logspace(2, 10, 50) 
Theta0, Theta1 = np.meshgrid(theta0, theta1)

LML = [[gp.log_marginal_likelihood(np.log([1.0, Theta0[i, j], Theta1[i, j]])) for i in range(Theta0.shape[0])] for j in range(Theta0.shape[1])]

print "Kom meg hit" 
LML = np.array(LML).T

vmin, vmax = (-LML).min(), (-LML).max()
print vmin, vmax
#vmax = 50
level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), 50), decimals=1)
plt.contour(Theta0, Theta1, -LML, levels=level, norm=LogNorm(vmin=vmin, vmax=vmax))
plt.colorbar()
plt.xlabel("Length scale Matern")
plt.ylabel("Length scale RBF")
plt.xscale("log")
plt.yscale("log")
plt.tight_layout()
plt.show()


"""
theta0 = np.asarray([0.001, 0.01, 0.1, 0.5, 1.0, 1.5, 2, 2.5, 3, 5])

LMLs = np.zeros(len(theta0))

for i in range(len(theta0)):
    print "HEY", i
    mykernel =  C(10, (1e-3, 1e3))*Matern(np.asarray([10, 0.1]), (1e-2, 1e6), nu=theta0[i]) + C(0.001, (1e-6,1e2))
    gp = GaussianProcessRegressor(kernel=mykernel, random_state=42).fit(X_train, y_train)
    print gp.log_marginal_likelihood()
    print gp.kernel_
    LMLs[i] = gp.log_marginal_likelihood()

plt.plot(theta0, np.asarray(LMLs))

plt.show()
"""
