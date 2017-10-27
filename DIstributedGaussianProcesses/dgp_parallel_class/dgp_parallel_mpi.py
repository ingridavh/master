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

#Import pandas for data formatting
import pandas as pd

#Import SciKit-learn modules
import sklearn as sk
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.cross_validation import train_test_split

# Import sklearns parallelization tool

from sklearn.externals.joblib import parallel






# Import MPI for Python for parallelizing
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    print "The number of cores is", size

####################################################
# class DGP                                        #
####################################################


class dgp_parallel:
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


    def fit(self, X, y, trainsize=0.1, log_true=True, alpha=0.001):
        data = X
        target = y
        

        if self.allow_printing == True:
            print "Data dimensions: ", data.shape
            print "Target dimensions: ", target.shape

        # Split data into training and test data
        X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=self.random_state, train_size = trainsize)
        
        # Change data to log
        y_train = np.log10(y_train)
        y_test = np.log10(y_test)


        # Calculate the number of experts each core should use
        if self.n_experts%size == 0:
            n_local = self.n_experts/size
        else:
            print "Error! Number of experts must be a multiple of the number of cores"
        h = 1

        local_min = rank*n_local*h
        local_max = local_min + n_local*h
            
        ############################################################
        # DGP                                                      #
        ############################################################

        # Divide and distribute data to experts 

        n_experts = self.n_experts
        
        n_points = len(data[:,])
        len_subsets = n_points/n_experts

        subsets_X = np.array_split(X_train, n_experts)
        subsets_y = np.array_split(y_train, n_experts)

        # Arrays to save temporary mus and sigmas

        mus_experts = np.zeros(( n_experts, len(X_test) ))
        sigmas_experts = np.zeros(( n_experts, len(X_test) ))
        prior_covs = np.zeros(( n_experts, len(X_test) ))

        # Make Gaussian fit for each expert

        for expert_id in range(local_min, local_max, h):
            if self.allow_printing == True:
                print "Expert number %i reporting for duty" % int(expert_id+1)
                print "Fitting kernel to data..."
                
            mus_, sigmas_, prior_covs_ = self.fit_my_expert(expert_id, subsets_X[expert_id], subsets_y[expert_id], X_test, alpha )
            mus_experts += mus_
            sigmas_experts += sigmas_
            prior_covs += prior_covs_
                
      
        # Define the 'totals', where sums should be sent to rank=0
        # These are characterized by an underscore at the end, e.g. total_
        mus_experts_tot = np.zeros(( n_experts, len(X_test) ))
        sigmas_experts_tot = np.zeros(( n_experts, len(X_test) ))
        prior_covs_tot = np.zeros(( n_experts, len(X_test) ))

        # Each expert sends its result to rank=0
        comm.Reduce(mus_experts, mus_experts_tot, op=MPI.SUM, root=0)
        comm.Reduce(sigmas_experts, sigmas_experts_tot, op=MPI.SUM, root=0)
        comm.Reduce(prior_covs, prior_covs_tot, op=MPI.SUM, root=0)
        
        if rank == 0:

            sigma_rbcm_neg = np.zeros(len(X_test))
            mu_rbcm = np.zeros(len(X_test))
            # Fill in sigma values to get the total sum
            for expert_id in range(n_experts):

                sigma_rbcm_neg_fill = self.fill_my_sigmas(expert_id, X_test, mus_experts_tot, sigmas_experts_tot, prior_covs_tot)
                sigma_rbcm_neg += sigma_rbcm_neg_fill
                
            # Fill in mu values using the sum of sigma
            for expert_id in range(n_experts):
                # Calculate mu_rbcm

                mu_rbcm_fill = self.fill_my_mus(expert_id, X_test, mus_experts_tot, sigmas_experts_tot, prior_covs_tot, sigma_rbcm_neg)
                mu_rbcm += mu_rbcm_fill

            # Calculate errors
            rel_err = (10**y_test -10**mu_rbcm)/10**y_test

            print rel_err
            
            self.write_results(rel_err, mu_rbcm, sigma_rbcm_neg**(-1))
           

    def fit_my_expert(self, expert_id, subset_X, subset_y, X_test, alpha):
        #########################################################################
        # Function that fits and predicts for a given expert.                   #
        # Returns mus, sigmas and prior covariances of the same size as the     #
        # final arrays, to be added to these.                                   #
        #########################################################################

        N = len(X_test)

        mus = np.zeros(( self.n_experts, N ))
        sigmas = np.zeros(( self.n_experts, N ))
        prior_covs = np.zeros(( self.n_experts, N ))
        
        gp = GaussianProcessRegressor(kernel=self.kernel, alpha=alpha, normalize_y = False, n_restarts_optimizer = 0).fit(subset_X, subset_y)

        if self.allow_printing == True:
            print "Expert number %i finished fitting!" % int(expert_id+1)
            print "Expert number %i 's kernel parameters: " % int(expert_id+1), gp.kernel_
                
        # Fill arrays with predictions of means and sigmas for each test point in y_test
        for k in range(N):
            my_X = X_test[k].reshape(1,-1) #Reshape 1D-array for .predict
            mu_star, sigma_star_mean = gp.predict(my_X, return_cov=True)
            prior_cov = self.kernel(X_test[k])
                
            mus[expert_id][k] = mu_star
            sigmas[expert_id][k] = sigma_star_mean
            prior_covs[expert_id][k] = prior_cov

        return mus, sigmas, prior_covs




    
    def fill_my_sigmas(self, expert_id, X_test, mus_experts, sigmas_experts, prior_covs):
        #########################################################################
        # Do distributed Gaussian process to estimate the value of              #
        # sigma for each test point.                                            #
        #########################################################################

        N = len(X_test)

        sigma_rbcm_neg = np.zeros(N)
        
        for k in range(N):
            mu_star = mus_experts[expert_id][k]
            sigma_star_mean = sigmas_experts[expert_id][k]
            prior_cov = prior_covs[expert_id][k]
            
            beta = 0.5*(np.log(prior_cov)-np.log(sigma_star_mean))
            sigma_rbcm_neg[k] = beta*sigma_star_mean**(-1)+(1./self.n_experts - beta)*prior_cov**(-1)

        # Return estimate of covariances
        return sigma_rbcm_neg






    
    def fill_my_mus(self, expert_id, X_test, mus_experts, sigmas_experts, prior_covs, sigma_rbcm_neg):
        #########################################################################
        # Do distributed Gaussian process to estimate the value of              #
        # mu for each test point, using the sum of sigmas.                      #
        #########################################################################

        N = len(X_test)

        mu_rbcm = np.zeros(N)

        for k in range(N):
            mu_star = mus_experts[expert_id][k]
            sigma_star_mean = sigmas_experts[expert_id][k]
            prior_cov = prior_covs[expert_id][k]

            beta = 0.5*(np.log(prior_cov)-np.log(sigma_star_mean))
            mu_rbcm[k] =  sigma_rbcm_neg[k]**(-1)*(beta*sigma_star_mean**(-1)*mu_star)

        # Return estimate of means
        return mu_rbcm

        

    
            
    def write_results(self, rel_err, mus, sigmas):
        # Write results to file

        outfile = open(self.output_name, 'w')
        
        outfile.write('Test results: DGP Error -- Mus -- Sigmas \n')
        for i in range(len(rel_err)):
            outfile.write(str(rel_err[i])+' ')
            outfile.write(str(mus[i])+ ' ')
            outfile.write(str(sigmas[i])+' ')
            outfile.write('\n')

        outfile.close()

        if self.allow_printing == True:
            print "Results are found in file", outfile

        

