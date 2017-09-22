"""
Implement and test the Distributed Gaussian Processes using
SciKitLearn.

Compares with full GP for small datasets.

@author: Ingrid A V Holm
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.stats import norm
import scipy.stats as stats
import matplotlib.mlab as mlab

#Import pandas for data formatting
import pandas as pd

#Import SciKit-learn modules
import sklearn as sk
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_boston


#Import Seaborn for pretty plots
import seaborn as sns


#Define a simple function for benchmarking
def f_x(x):
    return 4*x[:,0]*x[:,1]
    
x_values = abs(np.random.randn(1000,2))*10
y_values = f_x(x_values)

if len(sys.argv) >= 2:
    #Set fraction of data to be used as training data
    trainsize = sys.argv[1]
else:
    trainsize = 0.1

if len(sys.argv) >= 3:
    #OBS! Fix this and read data from file! Pandas!!!!!!!!!!!!!!!!!!!!!! 
    infile = sys.argv[2]
else:
    #Load toy data set
    #If return_X_y=True this returns (data, target) tuple
    infile = load_boston(return_X_y = True)
    print "No input file was given, so I'll use the toy data set 'boston'!"

if len(sys.argv) >= 4:
    #Read name of outfile from command line
    outfile = sys.argv[3]
else:
    outfile = "results.txt"

print "I will dump the results in file: ", outfile

#Make data ready for Gaussian Process

#data = infile[0]
#targets = infile[1]

data = x_values
targets = y_values

print "Input data dimensions: ", data.shape
print "Target shape: ", targets.shape

#Split into training and test data
X_train, X_test, y_train, y_test = train_test_split(data, targets, random_state=42, train_size=trainsize)

#Choose the number of data subsets
n_subsets = 8

#Choose kernel
kernel = ExpSineSquared(length_scale=10.0, periodicity=1.0, length_scale_bounds=(1e-02, 1000.0), periodicity_bounds=(1e-02, 1000.0))


############################################################
# Regular GP                                               #
############################################################

#Fit GP to training data
print "Fitting kernel to data..."
gp = GaussianProcessRegressor(kernel=kernel, alpha=1).fit(X_train, y_train)
print "Finished fitting!"

#Print mean and std
print "Initial kernel: ", kernel
print "Kernel after fit: ", gp.kernel_

#Predict for test data
y_predict, cov_y = gp.predict(X_test, return_cov=True)
y_predict, std_y = gp.predict(X_test, return_std=True)

#Compute relative errors
err_rel = (y_predict - y_test)/y_test

#plt.hist(err_rel, bins=30)
#plt.show()




############################################################
# DGP                                                      #
############################################################

n_points = len(data[:,])
len_subsets = n_points/n_subsets

subsets_X = np.array_split(X_train, n_subsets)
subsets_y = np.array_split(y_train, n_subsets)

kernel_params = []
means = np.zeros((n_subsets, len(X_test)))
variances = np.zeros((n_subsets, len(X_test)))

for i in range(n_subsets):
    #print "This is expert number %i" % int(i+1)
    #print "Fitting kernel to data..."
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1).fit(subsets_X[i], subsets_y[i])
    #print "Finished fitting!"
    kernel_params.append(gp.kernel_)
    #print "Kernel parameters: ", gp.kernel_

    #Predict y-values using the test set, and save mean and variance
    y_predict_mean, y_predict_std = gp.predict(X_test, return_std=True)
    
    means[i] = y_predict_mean
    variances[i] = y_predict_std

    
#print "My kernels are: ", kernel_params




######################################################################
# Plot each prediction with error bar                                #
######################################################################

x = np.linspace(0,1,len(X_test))

"""
#plt.errorbar(x, y, yerr=yerr)
plt.fill_between(x, y-yerr, y+yerr)
plt.plot(x, y, '-', color='salmon')
#plt.plot(x, variances[0], x, variances[1], x, variances[2])
plt.title('Mean error with std', size='xx-large')
plt.ylabel('Mean error', size='large')
plt.xlabel('log(NLO)', size='large')
#plt.ylim(15,25)
plt.show()
"""
