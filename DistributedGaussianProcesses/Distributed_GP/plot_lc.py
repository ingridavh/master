import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, Matern
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import sys

import pandas as pd
if len(sys.argv) >= 2:
    myfile = sys.argv[1]

#####################################################
# Read data from file                               #
#####################################################

#df_lin = pd.read_csv(myfile, sep=" ", skiprows=1)
df_lin = pd.read_csv(myfile, sep=" ", header=None, skiprows=1, skipinitialspace=True)
df_lin.columns = ["Train size", "Train score mean", "Train score std", "Test score mean", "Test score std"]
#df_lin = df_lin.drop(["Unnamed: 4"], axis=1)

train_sizes = df_lin["Train size"].values.ravel()
train_scores_mean = df_lin["Train score mean"].values.ravel()
train_scores_std = df_lin["Train score std"].values.ravel()
test_scores_mean = df_lin["Test score mean"].values.ravel()
test_scores_std = df_lin["Test score std"].values.ravel()

######################################################
# Function that plots learning curve                 #
######################################################
plt.figure()
#plt.title('Gaussian Process')
plt.xlabel("Training examples")
plt.ylabel("Score")

#train_scores_mean = np.mean(train_scores, axis=1)
#train_scores_std = np.std(train_scores, axis=1)
#test_scores_mean = np.mean(test_scores, axis=1)
#test_scores_std = np.std(test_scores, axis=1)

#print "Train sizes: ", train_sizes
#print "Train scores: ", train_scores
#print "Test scores: ", test_scores

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
#plt.xscale('log')
#plt.ylim([0.98,1.001])
plt.ylim([0.89,1.01])
plt.ylim([0.9980, 1.0001])
#plt.savefig('/home/ingrid/Documents/Master/ML/cv/abel3.pdf')

plt.show()
