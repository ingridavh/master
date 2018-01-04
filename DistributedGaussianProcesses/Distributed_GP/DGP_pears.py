"""
Created on 30.08.2017

 Reads data files output from harvest_slha.py using pandas and performs ML
 regression with sklearn. Currently uses Gaussian Processes.

 @author Ingrid Angelica Vazquez Holm
"""

import numpy as np
import sys
import pandas as pd
import sklearn as sk
from sklearn.gaussian_process.kernels import RBF,  WhiteKernel, DotProduct, ExpSineSquared, ConstantKernel as C

from dgp import dgp

print('Numpy version ' + np.__version__)
print('Pandas version ' + pd.__version__)
print('Sklearn version ' + sk.__version__)

#Small number
eps = 1E-20 # Used to regularize points with zero cross section

#Training size
if len(sys.argv) >= 2:
    trainsize = float(sys.argv[1])
else:
    trainsize = 0.001

if len(sys.argv) >= 3:
    outfile = sys.argv[2]
else:
    print "Error! No outfile was provided."
    sys.exit(0)

if len(sys.argv) >= 4:
    my_kernel = str(sys.argv[3])
else:
    print "Error! No kernel was provided."
    sys.exit(1)

if len(sys.argv) >= 5:
    num_exp = int(sys.argv[4])
else:
    print "Error! The number of experts was not specified."
    sys.exit(2)

    
print "Size of the training set is %.3f of the total dataset." % trainsize

#################################################################
# Read in and treat data (mostly) using pandas.                 #
#################################################################

# Read in data (last column is BS)
data_lin = pd.read_csv('/home/ingrid/Documents/Master/ML/data/pears.dat', sep=' ', skipinitialspace=True)

#Create data frames in pandas
df_lin = pd.DataFrame(data_lin)

#Drop BS column
df_lin = df_lin.drop('Unnamed: 5', axis=1)

#Find zero cross sections and replace with small number
df_lin = df_lin.replace(0.0, eps)

#Check properties of data
print "len(data) = ", len(df_lin)
print "data.shape = ", df_lin.shape

# Define features to train on and target nubers (used below)
# Convert to array for scikit using values
# ravel is used to avoid [[n]], i.e. n x 1 arrays
feature_list = ["3.mSquark2L", "5.M3"]
n_features = len(feature_list)
target_list = ["2.qq_NLO"]
features = df_lin[feature_list].values
target = df_lin[target_list].values.ravel()


##############################################################
# Distributed Gaussian Processes                             #
##############################################################



# Define kernel to be used in GP
kernel1  =  C(1.0, (1e-2, 1e2)) * RBF(10, (1e-3, 1e3))
kernel2 = DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-05, 100.0))
kernel3 = ExpSineSquared(length_scale=10.0, periodicity=1.0, length_scale_bounds=(1e-02, 1000.0), periodicity_bounds=(1e-02, 1000.0))
kernel4 = RBF(1.0, (1e-3, 1e3)) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e+3))
kernel5 =  C(1.0, (1e-2, 1e2)) * RBF(10, (1e-4, 1e4)) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 1e+3))
kernel6 = C(1.0, (1e-2, 1e2)) * RBF(10, (1e-4, 1e4)) + C(0.001, (1e-6,1e-1))
kernel7 = C(1.0, (1e-2, 1e2)) * RBF(10, (1e-4, 1e4)) + C(0.001, (1e-6,1e-1))*WhiteKernel(0.01, (1e-5, 1e-1))


# Set the wanted kernel
if my_kernel == 'kernel1':
    my_kernel = kernel1
elif my_kernel == 'kernel2':    my_kernel = kernel2
elif my_kernel == 'kernel3':
    my_kernel = kernel3
elif my_kernel == 'kernel4':
    my_kernel = kernel4
elif my_kernel == 'kernel5':
    my_kernel = kernel5
elif my_kernel == 'kernel6':
    my_kernel = kernel6
elif my_kernel == 'kernel7':
    my_kernel = kernel7
else:
    print "Error! Not a valid kernel."
    sys.exit(2)



print "Starting gaussian process analysis...\n"

my_dgp = dgp(num_exp, outfile, kernel=my_kernel)
(mu, sigma) = my_dgp.fit(features, target, trainsize=trainsize, alpha=0.01)
