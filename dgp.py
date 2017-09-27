"""
Class for Distributed Gaussian Processes. 
Input is: 

- Input file
- Output file name
- Number of experts
- Size of training data (fraction e.g. 0.1)
- Kernel

@author: Ingrid A V Holm
"""

"""
Implement and test the Distributed Gaussian Processes using
SciKitLearn.

Compares with full GP for small datasets.

@author: Ingrid A V Holm
"""
# Import some libraries

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
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_boston




####################################################
# class DGP                                        #
####################################################


class dgp:

    def __init__(self, input_name, output_name)

    # Initialize with input and output name
    self.infile = open(input_name)
    self.outfile = open(output_name, 'w')

    trainsize = 
    data = x_values
    targets = y_values



    X_train, X_test, y_train, y_test = train_test_split(data, targets, random_state=42, train_size=trainsize)

    # Change to log
    y_train = np.log10(y_train)
    y_test = np.log10(y_test)

    #Choose kernel
    kernel = RBF(length_scale=1, length_scale_bounds=(1e-02, 1e05)) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-03, 1000.0))

############################################################
# DGP                                                      #
############################################################

    n_points = len(data[:,])
    len_subsets = n_points/n_experts

    subsets_X = np.array_split(X_train, n_experts)
    subsets_y = np.array_split(y_train, n_experts)
    
    kernel_params = []

    sigma_rbcm_neg = np.zeros(len(X_test))
    mu_rbcm = np.zeros(len(X_test))

    mus_experts = np.zeros(( n_experts, len(X_test) ))
    sigmas_experts = np.zeros(( n_experts, len(X_test) ))
    prior_covs = np.zeros(( n_experts, len(X_test) ))


for i in range(n_subsets):
    print "Expert number %i reporting for duty" % int(i+1)
    print "Fitting kernel to data..."
    gp_temp = GaussianProcessRegressor(kernel=kernel, alpha=0.001, normalize_y = False, n_restarts_optimizer = 0).fit(subsets_X[i], subsets_y[i])
    print "Finished fitting!"
    kernel_params.append(gp_temp.kernel_)
    print "Kernel parameters: ", gp_temp.kernel_

    # Do BCM stuff
    for k in range(len(X_test)):
        my_X = X_test[k].reshape(1,-1)
        mu_star, sigma_star_mean = gp_temp.predict(my_X, return_cov=True)
        prior_cov = kernel(X_test[k])

        mus_experts[i][k] = mu_star
        sigmas_experts[i][k] = sigma_star_mean
        prior_covs[i][k] = prior_cov
        

for i in range(n_experts):
    # Calculate sigma_rbcm
    for k in range(len(X_test)):
        mu_star = mus_experts[i][k]
        sigma_star_mean = sigmas_experts[i][k]
        prior_cov = prior_covs[i][k]
        
        beta = 0.5*(np.log(prior_cov)-np.log(sigma_star_mean))
        sigma_rbcm_neg[k] += beta*sigma_star_mean**(-1)+(1./n_experts - beta)*prior_cov**(-1)


for i in range(n_experts):
    # Calculate mu_rbcm
    for k in range(len(X_test)):
        mu_star = mus_experts[i][k]
        sigma_star_mean = sigmas_experts[i][k]
        prior_cov = prior_covs[i][k]

        beta = 0.5*(np.log(prior_cov)-np.log(sigma_star_mean))
    
        mu_rbcm[k] +=  sigma_rbcm_neg[k]**(-1)*(beta*sigma_star_mean**(-1)*mu_star)
    

rel_err = (10**y_test -10**mu_rbcm)/10**y_test
rel_err_gp = (10**y_test - 10**y_predict)/10**y_test

########################################################
# Write results to file                                #
########################################################

outfile = open(outfile_name, 'w')
outfile.write('Test results: GP Error -- DGP Error -- Mus -- Sigmas \n')
for i in range(len(X_test)):
    outfile.write(str(rel_err_gp[i])+' ')
    outfile.write(str(rel_err[i])+' ')
    outfile.write(str(mu_rbcm[i])+ ' ')
    outfile.write(str(sigma_rbcm_neg[i])+' ')
    outfile.write('\n')

outfile.close()

print "Results are found in file", outfile_name

