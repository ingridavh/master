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
        
        subsets_X = np.array_split(X_train, n_experts+1)
        subsets_y = np.array_split(y_train, n_experts+1)

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

        mu_star = np.zeros((n_experts, N))
        sigma_star_mean = np.zeros((n_experts, N))
        prior_cov = np.zeros((n_experts, N))


        # Make a prediction for each expert, and save it in an array

        """
        for expert in range(n_experts):
            # This part should also be parallelized
            t1 = time.time()
            mu_star_, sigma_star_mean_ = out[expert].predict(X_test, return_std=True)

            mu_star[expert] = mu_star_
            sigma_star_mean[expert] = sigma_star_mean_**2 
            prior_cov[expert] = np.diag(self.kernel(X_test)) # Bytt til expert.kernel_ (?)
            #prior_cov[expert] = np.diag(out[expert].kernel_(X_test))

            print "Expert fit took the time ", time.time() - t1
        """

 

        # Parallelized routine for predict

        prediction = Parallel(n_jobs=self.njobs, verbose=4)(delayed(predict_my_expert)(out[i], X_test) for i in range(n_experts)) # mu_star, sigma_star_mean

    


        
        # Fill in sigma values to get the total sum
        # Parallelized routine for sigma fill
        for i in range(n_experts):

            mu_star = prediction[i][0]
            sigma_star_mean = prediction[i][1]

            sigma_rbcm_neg_fill = self.fill_my_sigmas(N, mu_star, sigma_star_mean, prior_cov)
            sigma_rbcm_neg += sigma_rbcm_neg_fill

        """
        # Fill in sigma values to get the total sum
        for i in range(n_experts):

            sigma_rbcm_neg_fill = self.fill_my_sigmas(N, mu_star[i], sigma_star_mean[i], prior_cov[i])
            sigma_rbcm_neg += sigma_rbcm_neg_fill
        """

            
        # Fill in mu values using the sum of sigma
        for i in range(n_experts):
            mu_rbcm_fill = self.fill_my_mus(N, mu_star[i], sigma_star_mean[i], prior_cov[i], sigma_rbcm_neg)
            mu_rbcm += mu_rbcm_fill

        msq = X_test[:, 0]
        mg = X_test[:, 1]
            
        # Calculate errors
        rel_err = ( 10**y_test - 10**mu_rbcm )/ 10**y_test

        # Write results to file
        self.write_results(rel_err, mu_rbcm, sigma_rbcm_neg, y_test, msq, mg)

    
    def fill_my_sigmas(self, N, mus, sigmas, priors):
        #########################################################################
        # Do distributed Gaussian process to estimate the value of              #
        # sigma for each test point.                                            #
        #########################################################################

        sigma_rbcm_neg = np.zeros(N)
        
        for k in range(N):
            mu_star = mus[k]
            sigma_star_mean = sigmas[k]
            prior_cov = priors[k]
            
            beta = 0.5*(np.log(prior_cov)-np.log(sigma_star_mean))
            sigma_rbcm_neg[k] = beta*sigma_star_mean**(-1)+(1./self.n_experts - beta)*prior_cov**(-1)

        # Return estimate of covariances
        return sigma_rbcm_neg


    
    def fill_my_mus(self, N, mus, sigmas, priors, sigma_rbcm_neg):
        #########################################################################
        # Do distributed Gaussian process to estimate the value of              #
        # mu for each test point, using the sum of sigmas.                      #
        #########################################################################

        mu_rbcm = np.zeros(N)

        for k in range(N):
            mu_star = mus[k]
            sigma_star_mean = sigmas[k]
            prior_cov = priors[k]

            beta = 0.5*(np.log(prior_cov)-np.log(sigma_star_mean))
            mu_rbcm[k] =  sigma_rbcm_neg[k]**(-1)*(beta*sigma_star_mean**(-1)*mu_star)

        # Return estimate of means
        return mu_rbcm

    
            
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

            

    def predict_my_expert(self, X_test, gp_expert):
        # Function that does a prediction for the ith expert

        mu_star, sigma_star_mean = gp_expert.predict(X_test, return_std=True)
        return mu_star, sigma_star_mean


    
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

