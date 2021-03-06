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
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_boston

####################################################
# class DGP                                        #
####################################################


class dgp:
    """
    An implementation of Disitributed Gaussian Processes as described by Deisenroth and Ng (2015). This class uses the GP packages from scikit-learn for each individual expert, and parallellizes the code.

    kernel : kernel object
        The kernel specifying the covariance function of the GP. If None is
        passed, the kernel "1.0 * RBF(1.0)" is used as default. Note that
        the kernel's hyperparameters are optimized during fitting.

    random_state : int, RandomState instance or None, optional (default: None)
        The generator used to initialize the centers.
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """


    
    def __init__(self, n_experts, output_name, kernel=ConstantKernel, n_restarts_optimizer=0, random_state=None,  allow_printing=True):

        # Initialize with input and output name
        self.n_experts = n_experts
        self.output_name = output_name
        self.kernel = kernel
        self.n_restarts_optimizer = n_restarts_optimizer
        self.random_state = random_state
        self.allow_printing = allow_printing

        if allow_printing == True:
            print "Created an instance of dgp with %i experts, and output file %s." % (n_experts, output_name)


    def fit(self, X, y, trainsize=0.1, alpha=0.001):
        data = X
        target = y

        if self.allow_printing == True:
            print "Data dimensions: ", data.shape
            print "Target dimensions: ", target.shape

        # Split data into training and test data
        X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=self.random_state, train_size = trainsize)
        
        # Change data to log if desired, default is True
        
        y_train = np.log10(y_train)
        y_test = np.log10(y_test)


        ############################################################
        # DGP                                                      #
        ############################################################

        # Divide and distribute data to experts 

        n_experts = self.n_experts
        
        n_points = len(data[:,])
        len_subsets = n_points/n_experts

        subsets_X = np.array_split(X_train, n_experts)
        subsets_y = np.array_split(y_train, n_experts)

        kernel_params = []

        # Final mu and sigma to be filled

        sigma_rbcm_neg = np.zeros(len(X_test))
        mu_rbcm = np.zeros(len(X_test))

        # Arrays to save temporary mus and sigmas

        mus_experts = np.zeros(( n_experts, len(X_test) ))
        sigmas_experts = np.zeros(( n_experts, len(X_test) ))
        prior_covs = np.zeros(( n_experts, len(X_test) ))


        # Make Gaussian fit for each expert

        for i in range(n_experts):
            if self.allow_printing == True:
                print "Expert number %i reporting for duty" % int(i+1)
                print "Fitting kernel to data..."

            # Gaussian process fit
            gp_temp = GaussianProcessRegressor(kernel=self.kernel, alpha=alpha, normalize_y = False, n_restarts_optimizer = 0).fit(subsets_X[i], subsets_y[i])

            if self.allow_printing == True:
                print "Expert number %i finished fitting!" % int(i+1)

            # Save optimized kernel parameters for this expert
            kernel_params.append(gp_temp.kernel_)
            
            if self.allow_printing == True:
                print "Expert number %i 's kernel parameters: " % int(i+1), gp_temp.kernel_

                
            # Fill arrays with predictions of means and sigmas for each test point in y_test
            for k in range(len(X_test)):
                my_X = X_test[k].reshape(1,-1) #Reshape 1D-array for .predict
                mu_star, sigma_star_mean = gp_temp.predict(my_X, return_cov=True)
                prior_cov = self.kernel(X_test[k])
                
                mus_experts[i][k] = mu_star
                sigmas_experts[i][k] = sigma_star_mean
                prior_covs[i][k] = prior_cov

        # Fill in sigma values to get the total sum
        for i in range(n_experts):
            # Calculate sigma_rbcm
            for k in range(len(X_test)):
                mu_star = mus_experts[i][k]
                sigma_star_mean = sigmas_experts[i][k]
                prior_cov = prior_covs[i][k]
        
                beta = 0.5*(np.log(prior_cov)-np.log(sigma_star_mean))
                sigma_rbcm_neg[k] += beta*sigma_star_mean**(-1)+(1./n_experts - beta)*prior_cov**(-1)


        # Fill in mu values using the sum of sigma

        for i in range(n_experts):
            # Calculate mu_rbcm
            for k in range(len(X_test)):
                mu_star = mus_experts[i][k]
                sigma_star_mean = sigmas_experts[i][k]
                prior_cov = prior_covs[i][k]

                beta = 0.5*(np.log(prior_cov)-np.log(sigma_star_mean))
    
                mu_rbcm[k] +=  sigma_rbcm_neg[k]**(-1)*(beta*sigma_star_mean**(-1)*mu_star)


        # Calculate errors

        rel_err = (10**y_test -10**mu_rbcm)/10**y_test
        
        m1 = X_test[:, 0]
        m2 = X_test[:, 1]

        y_test_out = 10**y_test
            
        self.write_results(rel_err, mu_rbcm, sigma_rbcm_neg, y_test_out, m1, m2)

        # Return predicted values
        return mu_rbcm, sigma_rbcm_neg
        

    def write_results(self, rel_err, mus, sigmas, y_test, m1, m2):
        # Write results to file

        outfile = open(self.output_name, 'w')
        
        outfile.write('Test results: DGP Error -- Mus -- Sigmas -- y_test -- x[0] -- x[1]\n')
        for i in range(len(rel_err)):
            outfile.write(str(rel_err[i])+' ')
            outfile.write(str(mus[i])+ ' ')
            outfile.write(str(sigmas[i])+' ')
            outfile.write(str(y_test[i])+' ')
            outfile.write(str(m1[i])+' ')
            outfile.write(str(m2[i])+' ')
            outfile.write('\n')

        outfile.close()

        if self.allow_printing == True:
            print "Results are found in file", outfile

        

