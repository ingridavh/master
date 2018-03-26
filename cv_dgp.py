"""
Implements k-fold cross validation for distributed gaussian processes.
- A model is trained using k-1 of the folds as training data.
- The resulting model is validated on the remaining part of the data.
The performance measure is then the average of the values computed in the loop.

@author: Ingrid A V Holm
"""

import numpy as np
import matplotlib as mpl
from dgp_parallel import dgp_parallel
import sklearn

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
import pandas as pd

mpl.style.use('ingrid_style')

################################
# Data handling
################################

# Read in data (last column is BS)
data_lin = pd.read_csv('/home/ingrid/Documents/Master/ML/Abel_all_crossections/data_40k.dat', sep=' ', skipinitialspace=True)

#Create data frames in pandas
df_lin = pd.DataFrame(data_lin)

# Lower cut at 10**-16
mask4 = df_lin['17.dLdL_NLO'] > 1e-16
df_lin = df_lin[mask4]

#Check properties of data
print "len(data) = ", len(df_lin)
print "data.shape = ", df_lin.shape

# Define features to train on and target nubers (used below)
# Convert to array for scikit using values
# ravel is used to avoid [[n]], i.e. n x 1 arrays
mean_list = ["39.mcL", "40.mdL", "41.mdR", "42.muL", "43.msL", "44.muR", "45.msR", "46.mcR"]
feature_list = ["38.mGluino", "40.mdL"]

means = df_lin[mean_list].mean(axis=1).values.ravel()

n_features = len(feature_list)
target_list = ["17.dLdL_NLO"]
features = df_lin[feature_list].values
target = df_lin[target_list].values.ravel()

long_features = np.zeros(( len(features), 3 ))
long_features[:,0] = features[:,0]
long_features[:,1] = features[:,1]
long_features[:,2] = means[:,]

target_m2 = target/features[:,0]**2

kernel_matern_3 = C(10, (1e-3, 1000)) * Matern(np.array([1000, 1000, 1000]), (1e3, 1e6), nu=1.0)

###############################
# End of data handling
###############################

N_experts = 4
R2_scores_cv_test_mean = np.zeros(N_experts)
R2_scores_cv_train_mean = np.zeros(N_experts)
R2_scores_cv_test_std = np.zeros(N_experts)
R2_scores_cv_train_std = np.zeros(N_experts)

N_per_expert = 200

n_experts = 1
my_njobs = 1

train_sizes = []
k = 5

for i in range(N_experts):

    trainsize = N_per_expert*(i+1)
    tot_size = int(float(k)/(k-1)*trainsize)

    train_sizes.append(trainsize)
    
    print "Size of training set: ", trainsize
    print "Size of test set: ", tot_size/k

    X, X_o, y, y_o = train_test_split(long_features, target_m2, random_state=42, train_size = tot_size)
    kf = KFold(n_splits=k, random_state=42)

    
    n_experts = i+1
    my_njobs = i+1
    
    r2_scores_test = []
    r2_scores_train = []
    
    for train_index, test_index in kf.split(X):
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        print len(X_train), len(X_test)

        my_dgp = dgp_parallel(n_experts, output_name=False, kernel=kernel_matern_3, verbose=False, njobs=my_njobs)
        X_test_dgp, y_test_dgp, mu_dgp_test, sigma_dgp_test, rel_err_dgp = my_dgp.fit_and_predict(long_features, target_m2, trainsize=trainsize, alpha=7.544e-07, X_train_ = X_train, X_test_ = X_test, y_train_ = y_train, y_test_ = y_test)

        print "len test", len(y_test_dgp)
        
        X_test_dgp, y_train_dgp, mu_dgp_train, sigma_dgp_train, rel_err_dgp = my_dgp.fit_and_predict(long_features, target_m2, trainsize=trainsize, alpha=7.544e-07, X_train_ = X_train, X_test_ = X_train, y_train_ = y_train, y_test_ = y_train)
        
        r2_scores_test.append(r2_score(y_test_dgp, mu_dgp_test) )
        print r2_score(y_test_dgp, mu_dgp_test)
        r2_scores_train.append(r2_score(y_train_dgp, mu_dgp_train) )

        
    r2_scores_test = np.asarray([r2_scores_test])
    R2_scores_cv_test_mean[i] = np.mean(r2_scores_test)
    R2_scores_cv_test_std[i] = np.std(r2_scores_test)

    r2_scores_train = np.asarray([r2_scores_train])
    R2_scores_cv_train_mean[i] = np.mean(r2_scores_train)
    R2_scores_cv_train_std[i] = np.std(r2_scores_train)
    
print R2_scores_cv_test_mean
print R2_scores_cv_test_std
print R2_scores_cv_train_mean
print R2_scores_cv_train_std

train_scores_mean = R2_scores_cv_train_mean
train_scores_std = R2_scores_cv_train_std
test_scores_mean = R2_scores_cv_test_mean
test_scores_std = R2_scores_cv_test_std

plt.figure()
plt.xlabel("Training examples")
plt.ylabel("Score")

plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
         label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
         label="Cross-validation score")

plt.legend(loc="best")

plt.show()
