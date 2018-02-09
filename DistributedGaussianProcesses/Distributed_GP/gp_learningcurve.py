import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, Matern
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

import pandas as pd
#Small number
eps = 1E-32 # Used to regularize points with zero cross section

# Read in data (last column is BS)
data_lin = pd.read_csv('/home/ingrid/Documents/Master/ML/Abel_test_params/lin_1.dat', sep=' ', skipinitialspace=True)

#Create data frames in pandas
df_lin = pd.DataFrame(data_lin)

#Drop BS column
df_lin = df_lin.drop('Unnamed: 5', axis=1)

#Find zero cross sections and replace with small number
#df_lin = df_lin.replace(0.0, eps)
mask = df_lin['2.qq_NLO'] != 0
df_lin = df_lin[mask]

#Check properties of data
print "len(data) = ", len(df_lin)
print "data.shape = ", df_lin.shape

# Define features to train on and target nubers (used below)
# Convert to array for scikit using values
# ravel is used to avoid [[n]], i.e. n x 1 arrays
feature_list = ["4.mGluino", "5.mcL"]
n_features = len(feature_list)
target_list = ["2.qq_NLO"]
features = df_lin[feature_list].values
target = df_lin[target_list].values.ravel()

# Pick out the first 1000 points
features_short = features[0:1000]
target_short = target[0:1000]

target_m2 = target/features[:,0]**2
target_short_m2 = target_short/features_short[:,0]**2

target = np.log10(target)
target_short = np.log10(target_short)
target_m2 = np.log10(target_m2)
target_short_m2 = np.log10(target_short_m2)

# Check data quality for features
feature1 = features[:,0]
feature2 = features[:,1]
plt.scatter(feature1, feature2)
plt.xlabel('feature1')
plt.ylabel('feature2')
plt.title('Data quality (Feature distribution)')
plt.show()

# Check that zeros are removed
plt.scatter(features_short[:,1], target_short)
plt.title("Cross section (true)")
plt.xlabel(r"$m_{\tilde{q}}$")
plt.ylabel(r"$\sigma$")
plt.show()

######################################################
# Function that plots learning curve                 #
######################################################

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    print "Train sizes: ", train_sizes
    print "Train scores: ", train_scores
    print "Test scores: ", test_scores
    
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
    return plt

cv = ShuffleSplit(n_splits=20, test_size=0.2, random_state=0)
title = "Learning Curves (Gaussian Processes)"
kernel = C(30) * Matern(nu=1.0) + C(50)
kernel_rbf = C(30) * RBF() + C(10)
#train_sizes = np.asarray([100,250,400,600,700, 800,1000, 1200, 2000])
estimator = GaussianProcessRegressor(kernel=kernel, alpha=0.01)
plot_learning_curve(estimator, title, features_short, target_short_m2, cv=cv, n_jobs=4, train_sizes=np.asarray([500, 600, 700, 800]))
plt.savefig('/home/ingrid/Documents/Master/ML/cv/lc_mat1_nozerom2_20n_mysizes.pdf')

plt.show()
