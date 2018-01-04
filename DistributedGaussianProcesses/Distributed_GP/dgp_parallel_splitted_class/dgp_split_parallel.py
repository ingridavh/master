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
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.stats import norm
import scipy.stats as stats
import matplotlib.mlab as mlab
import time

#Import pandas for data formatting
import pandas as pd

#Import SciKit-learn modules
import sklearn as sk
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.cross_validation import train_test_split

# Import sklearns parallelization tool
from sklearn.externals.joblib import Parallel, delayed

####################################################
# class DGP                                        #
####################################################

class dgp_split_parallel:
    """
    An implementation of Disitributed Gaussian Processes as described by 
    Deisenroth and Ng (2015). This class uses the GP packages from 
    scikit-learn for each individual expert, and parallellizes the code.

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
    
    def __init__(self, n_experts, output_name, kernel=ConstantKernel, n_restarts_optimizer=0, random_state=None, verbose=False, njobs=1):

        # Initialize with input and output name
        self.n_experts = n_experts
        self.output_name = output_name
        self.kernel = kernel
        self.n_restarts_optimizer = n_restarts_optimizer
        self.random_state = random_state
        self.verbose = verbose
        self.njobs = njobs

        if verbose == True:
            print "Created an instance of dgp with %i experts, and output file %s." % (n_experts, output_name)


    def fit(self, X_train, y_train, log_true=True, alpha=0.001):

        
        # Change data to log
        if log_true == True:
            y_train = np.log10(y_train)
            
        ############################################################
        # DGP                                                      #
        ############################################################

        # Divide and distribute data to experts 
        n_experts = self.n_experts

        # Hvorfor n_experts+1 her? 
        subsets_X = np.array_split(X_train, n_experts)
        subsets_y = np.array_split(y_train, n_experts)

        # Make Gaussian fit for each expert
        
        out = Parallel(n_jobs=self.njobs, verbose=4)(delayed(fit_my_expert)(subsets_X[i], subsets_y[i], alpha, n_experts, kernel=self.kernel) for i in range(n_experts))

        self.out = out


     
    def predict(self, X_test, y_test, log_true=True, alpha=0.001):

        # Change data to log
        if log_true == True:
            y_test = np.log10(y_test)
        
        N = len(X_test)
        n_experts = self.n_experts
        out = self.out
        
        sigma_rbcm_neg = np.zeros(N)
        mu_rbcm = np.zeros(N)

        # Set prior covariance points
        prior_cov = np.zeros(N)
        for k in range(N):
            my_X = X_test[k].reshape(1,-1)
            prior_cov[k] = self.kernel(my_X)
            

        # Make a prediction for each expert, and save it in an array
        prediction = Parallel(n_jobs=self.njobs, verbose=4)(delayed(predict_my_expert)(out[i], X_test) for i in range(n_experts)) # mu_star, sigma_star_mean

        prediction = np.array(prediction)
        mu_star = prediction[:,0]
        sigma_star_mean = prediction[:,1]
        
        # Fill in sigma values to get the total sum
        for i in range(n_experts):
            sigma_rbcm_neg_fill = self.fill_my_sigmas(out[i], mu_star[i], sigma_star_mean[i], prior_cov, X_test)
            sigma_rbcm_neg += sigma_rbcm_neg_fill

            
        # Fill in mu values using the sum of sigma
        for i in range(n_experts):            
            mu_rbcm_fill = self.fill_my_mus(out[i], mu_star[i], sigma_star_mean[i], prior_cov, X_test, sigma_rbcm_neg)
            mu_rbcm += mu_rbcm_fill
        


        msq = X_test[:, 0]
        mg = X_test[:, 1]
            
        # Calculate errors
        rel_err = ( 10**y_test - 10**mu_rbcm )/ 10**y_test

        # Write results to file
        self.write_results(rel_err, mu_rbcm, sigma_rbcm_neg, y_test, msq, mg)

        return mu_rbcm, sigma_rbcm_neg # Maa sigma inverteres?
    
    def write_results(self, rel_err, mus, sigmas, y_test, msq, mg):
        # Write results to file

        outfile = open(self.output_name, 'w')
        
        outfile.write('Test results: DGP Error -- Mus -- Sigmas -- Y_test -- x[0] -- x[1]\n')
        for i in range(len(rel_err)):
            outfile.write(str(rel_err[i])+' ')
            outfile.write(str(mus[i])+ ' ')
            outfile.write(str(sigmas[i])+' ')
            outfile.write(str(y_test[i])+' ')
            outfile.write(str(msq[i])+' ')
            outfile.write(str(mg[i])+' ')
            outfile.write('\n')

        outfile.close()

        if self.verbose == True:
            print "Results are found in file", outfile

            
# External functions to be parallelized

    def fill_my_sigmas(self, out, mus, sigmas, prior_covs, X_test):
        #########################################################################
        # Do distributed Gaussian process to estimate the value of              #
        # sigma for each test point.                                            #
        #########################################################################

        N = len(X_test)
        sigma_rbcm_neg = np.zeros(N)

        for k in range(N):
            mu_star = mus[k]
            sigma_star_mean = sigmas[k]
            #prior_cov = prior_covs[k]
            prior_cov = out.kernel_(X_test[k])

            beta = 0.5*(np.log(prior_cov)-np.log(sigma_star_mean))
            sigma_rbcm_neg[k] = beta*sigma_star_mean**(-1)+(1./self.n_experts - beta)*prior_cov**(-1)

        # Return estimate of covariances
        return sigma_rbcm_neg


    def fill_my_mus(self, out, mus, sigmas, prior_covs, X_test, sigma_rbcm_neg):
        #########################################################################
        # Do distributed Gaussian process to estimate the value of              #
        # mu for each test point, using the sum of sigmas.                      #
        #########################################################################

        N = len(X_test)
        mu_rbcm = np.zeros(N)

        for k in range(N):
            mu_star = mus[k]
            sigma_star_mean = sigmas[k]
            #prior_cov = prior_covs[k]
            prior_cov = out.kernel_(X_test[k])

            beta = 0.5*(np.log(prior_cov)-np.log(sigma_star_mean))
            mu_rbcm[k] =  sigma_rbcm_neg[k]**(-1)*(beta*sigma_star_mean**(-1)*mu_star)

        # Return estimate of means
        return mu_rbcm

    
def fit_my_expert(subset_X, subset_y, alpha, n_experts, kernel):
    #########################################################################
    # Function that fits and predicts for a given expert.                   #
    # Returns mus, sigmas and prior covariances of the same size as the     #
    # final arrays, to be added to these.                                   #
    #########################################################################

    t2 = time.time()    
    gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y = False, n_restarts_optimizer = 0).fit(subset_X, subset_y)

    print "Delta Time GP fit", time.time() - t2    
    print "Kernel parameters: ", gp.kernel_
                
    return gp


def predict_my_expert(gp_expert, X_test):
    # Function that does a prediction for the ith expert

    N = len(X_test)
    mu_star = np.zeros(N)
    sigma_star_mean = np.zeros(N)

    for k in range(N):
        my_X = X_test[k].reshape(1,-1)
        mu_star_, sigma_star_mean_ = gp_expert.predict(my_X, return_cov=True)
        sigma_star_mean[k] = sigma_star_mean_
        mu_star[k] = mu_star_
        
    return mu_star, sigma_star_mean



