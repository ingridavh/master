"""
Created on 30.08.2017

 Reads data files output from harvest_slha.py using pandas and performs ML
 regression with sklearn. Currently uses Gaussian Processes.

 @author Ingrid Angelica Vazquez Holm
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import ensemble
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,  WhiteKernel, DotProduct, ExpSineSquared, ConstantKernel as C

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

print "Size of the training set is %.3f of the total dataset." % trainsize

#################################################################
# Read in and treat data (mostly) using pandas.                 #
#################################################################

# Read in data (last column is BS)
data_lin = pd.read_csv('data_handling/apples.dat', sep=' ', skipinitialspace=True)

#Create data frames in pandas
df_lin = pd.DataFrame(data_lin)

#Drop BS column
df_lin = df_lin.drop('Unnamed: 5', axis=1)

#Find zero cross sections and replace with small number
df_lin = df_lin.replace(0.0, eps)

#Check properties of data
print "len(data) = ", len(df_lin)
print "data.shape = ", df_lin.shape
#print df_lin.columns
#print df_lin[0:5]

#Split data for training using function from sklearn (can also use pandas functionality)
train, test = train_test_split(df_lin, random_state=42, train_size=trainsize)

print "len(data) = ", len(train), len(test)
print "data.shape = ", train.shape, test.shape

# Define features to train on and target nubers (used below)
# Convert to array for scikit using values
# ravel is used to avoid [[n]], i.e. n x 1 arrays
feature_list = ["3.mSquark2L", "5.M3"]
n_features = len(feature_list)
target_list = ["2.qq_NLO"]
features = train[feature_list].values
target = train[target_list].values.ravel()
#print "Features: ", features
#print "Feature type: ", type(features)
#print "Target: ", target
features_test = test[feature_list].values
target_test = test[target_list].values.ravel()

#Change targets to log
target = np.log10(target)
target_test = np.log10(target_test)

##############################################################
# Gaussian Processes                                         #
##############################################################

print "Starting gaussian process analysis...\n"

# Define kernel to be used in GP
#kernel = 1.0 * RBF(length_scale=10.0, length_scale_bounds=(1e-2, 1e3))
kernel  =  C(1.0, (1e-2, 1e2)) * RBF(10, (1e-3, 1e3))
kernel2 = DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-05, 100.0))
kernel3 = ExpSineSquared(length_scale=10.0, periodicity=1.0, length_scale_bounds=(1e-02, 1000.0), periodicity_bounds=(1e-02, 1000.0))
kernel4 = RBF(936, (1e-2, 1e5)) + WhiteKernel(noise_level=0.374, noise_level_bounds=(1e-10, 1e+2))

# Free errors through white noise

print "Fitting the model to the data... \n"
gp = GaussianProcessRegressor(kernel=kernel4, alpha=0, normalize_y=False, n_restarts_optimizer=5).fit(features, target)

# Predict results
target_predict, cov_ = gp.predict(features_test, return_cov=True)
#Relative error

rel_diff = (np.power(10.,target_test) - np.power(10.,target_predict)) / np.power(10.,target_test)

#Write results to file
#Test features -- Predicted target -- std -- (Predicted target - Test target)

print "Writing results to file..."

writefile = open('bigapples.txt', 'w')
writefile.write('Test features:"3.mSquark2L" -- "5.M3" -- Predicted target -- Test_target -- Error \n')
for i in range(len(features_test)):
    writefile.write(str(features_test[i][0])+' ')
    writefile.write(str(features_test[i][1])+' ')
    writefile.write(str(target_predict[i])+' ')
    writefile.write(str(target_test[i])+' ')
    writefile.write(str(rel_diff[i])+' ')
    writefile.write('\n')

writefile.close()

print "Results are found in the file apple_seeds.txt."

#Write parameters for best fit to file

# Print likelihoods
print 'NLO   ', gp.log_marginal_likelihood(gp.kernel_.theta)
print 'Initial kernel:', kernel4
print 'Optimum kernel: ', gp.kernel_

"""
plt.hist(rel_diff, bins = np.linspace(-20,5, 100))
plt.title('Relative difference between predicted and test target values')
plt.xlabel('rel diff')
plt.ylabel('n')
plt.savefig('rel_diff_hist.pdf')
"""

