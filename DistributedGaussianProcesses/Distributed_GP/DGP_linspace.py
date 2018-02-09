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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.cross_validation import train_test_split
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
"""
if len(sys.argv) >= 3:
    outfile = sys.argv[2]
else:
    print "Error! No outfile was provided."
    sys.exit(0)
"""

print "Size of the training set is %.3f of the total dataset." % trainsize

#################################################################
# Read in and treat data (mostly) using pandas.                 #
#################################################################

# Read in data (last column is BS)
data_lin = pd.read_csv('/home/ingrid/Documents/Master/ML/Abel_long/data_290k.dat', sep=' ', skipinitialspace=True)

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

print "Max target: ", max(np.log10(target))
print "Min target: ", min(np.log10(target))

target_m2 = target/features[:,0]**2
target_m2 = np.log10(target_m2)

##############################################################
# Distributed Gaussian Processes                             #
##############################################################

# Define kernel to be used in GP
kernel_matern1 = C(10, (1e-3, 100)) * Matern(np.array([10, 10, 10]), (1e-4, 1e6), nu=1.0)

X_train, X_test, y_train, y_test = train_test_split(long_features, target_m2, random_state=42, train_size = trainsize)

# Fit Gaussian process
gp = GaussianProcessRegressor(kernel=kernel_matern1, alpha=7.544e-07, random_state=42)
gp.fit(X_train, y_train)

# Choose feature-values

N = 1000
mq_min = 10
mq_max = 4000

mg_test = np.zeros(N)+500
mq_test = np.linspace(mq_min, mq_max, N)
mean_test = (mg_test + 7000)/8. # mean = 1000

features_test = np.zeros((N, 3))
features_test[:,0] = mg_test
features_test[:,1] = mq_test
features_test[:,2]

# Chosen values
mus, sigmas = gp.predict(chosen_long_features, return_std=True)
mus += 2*np.log10(chosen_long_features[:,0])

N = len(mus)

mus_sort = np.zeros(N)
mq_sort = np.zeros(N)
sigmas_sort = np.zeros(N)
mq = chosen_long_features[:,1]

mq_sort_index = np.argsort(mq)
mus_sort = mus[mq_sort_index]
mq_sort = mq[mq_sort_index]
sigmas_sort = sigmas[mq_sort_index]

plt.scatter(chosen_long_features[:,1], mus)
plt.plot(mq_sort, mus_sort)
plt.xlabel(r'$m_{\tilde{q}}$')
plt.fill_between(mq_sort, mus_sort - sigmas_sort, mus_sort + sigmas_sort, alpha=0.5)
plt.show()
