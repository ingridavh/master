"""
For plotting the results of Distributed Gaussian Processes, 
calulated using the class dgp.py.

@author: Ingrid A V Holm
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.mlab as ml
import pandas as pd
import sys

# Import stats
from scipy import stats
from scipy.stats import norm

mpl.style.use('ingrid_style')

# Infile given from command line
if len(sys.argv) >= 2:
    infile = open(sys.argv[1])
else:
    infile = open('results.txt')

if len(sys.argv) >= 3:
    my_kernel = str(sys.argv[2])
    
if len(sys.argv) >= 4:
    matern = True
else:
    matern = False

df_lin = pd.read_csv(infile, sep=" ", header=None, skiprows=1, skipinitialspace=True)
df_lin.columns = ["Error", "Mus", "Sigmas", "Y_test", "Mg", "Mq", "Unnamed: 4"]
df_lin = df_lin.drop(["Unnamed: 4"], axis=1)

# Change back to cross section for sigma_m2
if matern == True:
    df_lin[["Y_test", "Mus"]] = df_lin[["Y_test", "Mus"]].add(2*np.log10(df_lin["Mg"]), axis="index")

errors = df_lin["Error"].values.ravel()
mus = df_lin["Mus"].values.ravel()
y_test = df_lin["Y_test"].values.ravel()
mg = df_lin["Mg"].values.ravel()
msq = df_lin["Mq"].values.ravel()

y_test = 10**y_test

N = len(y_test)

X, Y = np.meshgrid(mg[::500], msq[::500])
z = ml.griddata(mg[::500],msq[::500],  y_test[::500], X, Y, interp='linear')

g = plt.figure()
CS = plt.contour(X, Y, z)
plt.clabel(CS, inline=1)
plt.show()



