"""
A program that takes trained GP-models from joblib, and
combines them into a rBCM for prediction.
"""

import numpy as np
import sklearn as sk
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel, RBF
from sklearn.cross_validation import train_test_split
import pandas as pd
from sklearn.externals import joblib

class dgp_predict:
    """
    Combines Gaussian process experts into a 
    robust Bayesian Comittee Machine.
    """
    def __init__(self, models):
        self.models = models
        self.n_experts = len(models)

    def predict(self, X, kernel=None):
        """
        Takes as argument 'models': a list over models, 
        e.g. models=['uLuL_3000_1', 'uLuL_3000_2']

        and 'X': test features 

        and 'kernel': initial kernel used for all GPs
        """
        models = self.models
        self.X = X
        self.N = len(X)       
        N = len(X)
        n_experts = self.n_experts

        mus = np.zeros((n_experts, N))
        sigmas = np.zeros((n_experts, N))
        priors = np.zeros((n_experts, N))

        for j in range(n_experts):
            model = models[j]
            gp = joblib.load(model)

            if kernel == None:
                kernel = gp.kernel_

            mu_temp = np.zeros(N)
            sigma_temp = np.zeros(N)
            prior_temp = np.zeros(N)

            for k in range(N):
                x = X[k].reshape(1,-1)
                mu_temp[k], sigma_temp[k] = gp.predict(x, return_cov=True)
                prior_temp[k] = kernel(x)

            # Save predicted values
            mus[j] = mu_temp
            sigmas[j] = sigma_temp
            priors[j] = prior_temp

        sigma_rbcm_neg = np.zeros(N)
        mu_rbcm = np.zeros(N)

        for j in range(n_experts):
            sigma_rbcm_temp = self.fill_sigmas(X, mus[j], sigmas[j], priors[j])
            sigma_rbcm_neg += sigma_rbcm_temp

        self.sigma_rbcm_neg = sigma_rbcm_neg

        for j in range(n_experts):
            mu_rbcm_temp = self.fill_mus(X, mus[j], sigmas[j], priors[j])
            mu_rbcm += mu_rbcm_temp

        return mu_rbcm, np.sqrt(sigma_rbcm_neg**(-1))
        


    def fill_sigmas(self, X, mu, sigma, prior):
        N = self.N
        n_experts = self.n_experts

        sigma_rbcm_neg = np.zeros(N)

        for k in range(N):
            mu_star = mu[k]
            sigma_star_mean = sigma[k]
            prior_cov = prior[k]

            beta = 0.5*(np.log(prior_cov)-np.log(sigma_star_mean))
            sigma_rbcm_neg[k] = beta*sigma_star_mean**(-1)+(1./n_experts - beta)*prior_cov**(-1)

        return sigma_rbcm_neg

    def fill_mus(self, X, mu, sigma, prior):
        sigma_rbcm_neg = self.sigma_rbcm_neg
        N = self.N
        n_experts = self.n_experts

        mu_rbcm = np.zeros(N)

        for k in range(N):
            mu_star = mu[k]
            sigma_star_mean = sigma[k]
            prior_cov = prior[k]

            beta = 0.5*(np.log(prior_cov)-np.log(sigma_star_mean))
            mu_rbcm[k] =  sigma_rbcm_neg[k]**(-1)*(beta*sigma_star_mean**(-1)*mu_star)

            # Return estimate of means
        return mu_rbcm


