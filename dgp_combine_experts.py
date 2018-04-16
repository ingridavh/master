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

def fill_sigmas(X, mu, sigma, prior, n_experts):
    N = len(X)
    sigma_rbcm_neg = np.zeros(N)

    for k in range(N):
        mu_star = mu[k]
        sigma_star_mean = sigma[k]
        prior_cov = prior[k]
            
        beta = 0.5*(np.log(prior_cov)-np.log(sigma_star_mean))
        sigma_rbcm_neg[k] = beta*sigma_star_mean**(-1)+(1./n_experts - beta)*prior_cov**(-1)

    return sigma_rbcm_neg

def fill_mus(X, mu, sigma, prior, sigma_rbcm_neg, n_experts):

    N = len(X)
    mu_rbcm = np.zeros(N)

    
    for k in range(N):
        mu_star = mu[k]
        sigma_star_mean = sigma[k]
        prior_cov = prior[k]
        
        beta = 0.5*(np.log(prior_cov)-np.log(sigma_star_mean))
        mu_rbcm[k] =  sigma_rbcm_neg[k]**(-1)*(beta*sigma_star_mean**(-1)*mu_star)
        
        # Return estimate of means
    return mu_rbcm

def dgp_predict(models, X, kernel):
    """
    Takes as argument 'models': a list over models, 
    e.g. models=['uLuL_3000_1', 'uLuL_3000_2']

    and 'X': test features 

    and 'kernel': initial kernel used for all GPs
    """
    N = len(X)
    n_experts = len(models)

    mus = np.zeros((n_experts, N))
    sigmas = np.zeros((n_experts, N))
    priors = np.zeros((n_experts, N))

    print "OK 1"

    for j in range(n_experts):
        model = models[j]
        gp = joblib.load(model)

        mu_temp = np.zeros(N)
        sigma_temp = np.zeros(N)
        prior_temp = np.zeros(N)
        
        for k in range(N):
            x = X[k].reshape(1,-1)
            mu_temp[k], sigma_temp[k] = gp.predict(x, return_cov=True)
            #prior_temp = [gp.kernel_(X[k]) for k in range(N)]
            prior_temp[k] = kernel(x)
        
        # Save predicted values
        mus[j] = mu_temp
        sigmas[j] = sigma_temp
        priors[j] = prior_temp

    print "OK 2"
        
    sigma_rbcm_neg = np.zeros(N)
    mu_rbcm = np.zeros(N)
        
    for j in range(n_experts):
        sigma_rbcm_temp = fill_sigmas(X, mus[j], sigmas[j], priors[j], n_experts)
        sigma_rbcm_neg += sigma_rbcm_temp

    print "OK 3"
        
    for j in range(n_experts):
        mu_rbcm_temp = fill_mus(X, mus[j], sigmas[j], priors[j], sigma_rbcm_neg, n_experts)
        mu_rbcm += mu_rbcm_temp

    print "OK 4"
        
    return mu_rbcm, np.sqrt(sigma_rbcm_neg**(-1))

##############################################################
# Use function                                               #
##############################################################

from numpy.random import randn
import matplotlib.pyplot as plt

# Features
N_train = 400
N_test = 200
N = N_train + N_test

# Benchmark function

def f(x1, x2):
    return 4*x1+6*x2

np.random.seed(42)
x1 = randn(N) 
x2 = randn(N)
X = np.zeros((N,2))
X[:,0] = x1
X[:,1] = x2
fx = f(x1, x2)

X_train, X_test, y_train, y_test = train_test_split(X, fx, random_state=42, train_size = N_train)

X_train_1 = X_train[0:N_train/2]
y_train_1 = y_train[0:N_train/2]

X_train_2 = X_train[N_train/2:N_train]
y_train_2 = y_train[N_train/2:N_train]

kernel_rbf = C(10)*RBF(10, (1e-3, 1e3))

gp1 = GaussianProcessRegressor(kernel=kernel_rbf, random_state=42).fit(X_train_1, y_train_1)
gp2 = GaussianProcessRegressor(kernel=kernel_rbf, random_state=42).fit(X_train_2, y_train_2)

joblib.dump(gp1,'bm_200_1')
joblib.dump(gp1,'bm_200_2')

models = ['bm_200_1', 'bm_200_2']

# Distributed GP
mu_dgp, sigma_dgp = dgp_predict(models, X_test, kernel_rbf)
# Regular GP
mu = gp1.predict(X_test)

errors = (y_test - mu)/y_test
errors_dgp = (y_test - mu_dgp)/y_test

print np.mean(errors), np.std(errors)
print np.mean(errors_dgp), np.std(errors_dgp)

plt.scatter(X_test[:,0], y_test, color='blue', alpha=0.3, label='True')
plt.scatter(X_test[:,0], mu, color='red', alpha=0.3, label='GP')
plt.scatter(X_test[:,0], mu_dgp, color='green', alpha=0.3, label='DGP')
plt.legend()
plt.show()

#models = ['uLuL_1000_1', 'uLuL_1000_2']

