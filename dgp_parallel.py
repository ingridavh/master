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


class dgp_parallel:
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
    
    def __init__(self, n_experts, output_name, kernel=ConstantKernel, n_restarts_optimizer=0, random_state=42, verbose=False, njobs=1, eval_gradient=True, normalize_y=False, optimizer='fmin_l_bfgs_b'):

        # Initialize with input and output name
        self.n_experts = n_experts
        self.output_name = output_name
        self.kernel = kernel
        self.n_restarts_optimizer = n_restarts_optimizer
        self.random_state = random_state
        self.verbose = verbose
        self.njobs = njobs
        self.eval_gradient = eval_gradient
        self.normalize_y = normalize_y
        self.optimizer = optimizer

        if verbose == True:
            print "Created an instance of dgp with %i experts, and output file %s." % (n_experts, output_name)

    def fit_and_predict(self, X, y, trainsize=0.1, log_true=True, alpha=1e-10, X_train_=False, X_test_=False, y_train_ =False, y_test_= False):

        
        data = X
        target = y
        
        if self.verbose == True:
            print "Data dimensions: ", data.shape
            print "Target dimensions: ", target.shape

        # Split data into training and test data
        print data.shape, target.shape

        if X_train_.any() == False:
            X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=self.random_state, train_size = trainsize)

        else:
            X_train = X_train_
            X_test = X_test_
            y_train = y_train_
            y_test = y_test_

            
            
        print X_train.shape, X_test.shape

        # Reduce number of test points to 10k
        if len(y_test) > 20000:
            y_test = y_test[0:20000]
            X_test = X_test[0:20000]

        
        # Change data to log
        y_train = np.log10(y_train)
        y_test = np.log10(y_test)
            
        ############################################################
        # DGP                                                      #
        ############################################################

        # Divide and distribute data to experts 
        n_experts = self.n_experts
        
        subsets_X = np.array_split(X_train, n_experts)
        subsets_y = np.array_split(y_train, n_experts)
        
        N = len(X_test)

        # Make Gaussian fit for each expert
        
        out = Parallel(n_jobs=self.njobs, verbose=4)(delayed(fit_my_expert)(subsets_X[i], subsets_y[i], X_test, alpha, n_experts, kernel=self.kernel, normalize_y=self.normalize_y, optimizer=self.optimizer) for i in range(n_experts))
        
        mus = np.zeros((n_experts, N))
        sigmas = np.zeros((n_experts, N))
        priors = np.zeros((n_experts, N))
        gps = []
        
        for i in range(n_experts):
            mus[i] = out[i][0]
            sigmas[i] = out[i][1]
            priors[i] = out[i][2]
            gps.append(out[i][3])

        sigma_rbcm_neg = np.zeros(N)
        mu_rbcm = np.zeros(N)
        
        # Fill in sigma values to get the total sum
        for i in range(n_experts):
            sigma_rbcm_neg_fill = self.fill_my_sigmas(gps[i], X_test, mus[i], sigmas[i], priors[i])
            sigma_rbcm_neg += sigma_rbcm_neg_fill
        
        # Fill in mu values using the sum of sigma
        for i in range(n_experts):
            mu_rbcm_fill = self.fill_my_mus(gps[i], X_test, mus[i], sigmas[i], priors[i], sigma_rbcm_neg)
            mu_rbcm += mu_rbcm_fill


        mg = X_test[:, 0]
        msq = X_test[:, 1]
            
        # Calculate errors
        rel_err = ( 10**y_test - 10**mu_rbcm )/ 10**y_test

        # Write results to file
        if not self.output_name == False:
            self.write_results(rel_err, mu_rbcm, sigma_rbcm_neg, y_test, mg, msq)

        return X_test, y_test, mu_rbcm, sigma_rbcm_neg, rel_err 
    
    def fill_my_sigmas(self, out, X_test, mus, sigmas, priors):
        #########################################################################
        # Do distributed Gaussian process to estimate the value of              #
        # sigma for each test point.                                            #
        #########################################################################

        N = len(X_test)
        sigma_rbcm_neg = np.zeros(N)
        
        for k in range(N):
            mu_star = mus[k]
            sigma_star_mean = sigmas[k]
            #prior_cov = priors[k]
            prior_cov = out.kernel_(X_test[k])
            
            beta = 0.5*(np.log(prior_cov)-np.log(sigma_star_mean))
            sigma_rbcm_neg[k] = beta*sigma_star_mean**(-1)+(1./self.n_experts - beta)*prior_cov**(-1)

        # Return estimate of covariances
        return sigma_rbcm_neg


    
    def fill_my_mus(self, out, X_test, mus, sigmas, priors, sigma_rbcm_neg):
        #########################################################################
        # Do distributed Gaussian process to estimate the value of              #
        # mu for each test point, using the sum of sigmas.                      #
        #########################################################################

        N = len(X_test)

        mu_rbcm = np.zeros(N)

        for k in range(N):
            mu_star = mus[k]
            sigma_star_mean = sigmas[k]
            #prior_cov = priors[k]
            prior_cov = out.kernel_(X_test[k])

            beta = 0.5*(np.log(prior_cov)-np.log(sigma_star_mean))
            mu_rbcm[k] =  sigma_rbcm_neg[k]**(-1)*(beta*sigma_star_mean**(-1)*mu_star)

        # Return estimate of means
        return mu_rbcm

    
            
    def write_results(self, rel_err, mus, sigmas, y_test, mg, msq):
        # Write results to file

        outfile = open(self.output_name, 'w')
        
        outfile.write('Test results: DGP Error -- Mus -- Sigmas -- Y_test -- x[0] -- x[1]\n')
        for i in range(len(rel_err)):
            outfile.write(str(rel_err[i])+' ')
            outfile.write(str(mus[i])+ ' ')
            outfile.write(str(sigmas[i])+' ')
            outfile.write(str(y_test[i])+' ')
            outfile.write(str(mg[i])+' ')
            outfile.write(str(msq[i])+' ')
            outfile.write('\n')

        outfile.close()

        if self.verbose == True:
            print "Results are found in file", outfile

        

def fit_my_expert(subset_X, subset_y, X_test, alpha, n_experts, kernel, normalize_y, optimizer):
    #########################################################################
    # Function that fits and predicts for a given expert.                   #
    # Returns mus, sigmas and prior covariances of the same size as the     #
    # final arrays, to be added to these.                                   #
    #########################################################################

    N = len(X_test)
    mus = np.zeros(N)
    sigmas = np.zeros(N)
    prior_covs = np.zeros(N)
    
    if not isinstance(subset_X[0], np.ndarray):
        subset_X = subset_X.reshape(-1,1)

    
    gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y = normalize_y, n_restarts_optimizer = 0, optimizer=optimizer).fit(subset_X, subset_y)
 
    print "Kernel parameters: ", gp.kernel_
                
    # Fill arrays with predictions of means and sigmas for each test point in y_test
    
    for k in range(N): 
        my_X = X_test[k].reshape(1,-1) #Reshape 1D-array for .predict
        mu_star, sigma_star_mean = gp.predict(my_X, return_cov=True)
        prior_cov = kernel(X_test[k])
                    
        mus[k] = mu_star
        sigmas[k] = sigma_star_mean
        prior_covs[k] = prior_cov
        
    return mus, sigmas, prior_covs, gp
