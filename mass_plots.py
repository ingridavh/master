import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

mpl.style.use('ingrid_style')

data_lin = pd.read_csv('/home/ingrid/Documents/Master/ML/Abel_all_crossections/data_40k.dat', sep=' ', skipinitialspace=True)

#Create data frames in pandas
df_lin = pd.DataFrame(data_lin)

#Include outliers, but set to -32
eps = 1e-32
df_lin = df_lin.replace(0.0, eps)

mean_list = ["39.mcL", "40.mdL", "41.mdR", "42.muL", "43.msL", "44.muR", "45.msR", "46.mcR"]
feature_list = ["38.mGluino", "40.mdL"]

feature1 = "44.muR"
feature2 = "41.mdR"
target1 = "3.cLsL_NLO"

fig0 = plt.figure(0)
plt.scatter(df_lin[feature1].values.ravel(), df_lin[feature2].values.ravel())
plt.xlabel(r"$m_{\tilde{u}_L}$")
plt.ylabel(r"$m_{\tilde{d}_L}$")
plt.savefig('masses_muR_mdR.pdf')
fig0.show()
"""
fig1 = plt.figure(1)
plt.scatter(df_lin[feature1].values.ravel() , np.log10(df_lin[target1].values.ravel()/df_lin["38.mGluino"].values.ravel()**2), alpha=0.3)
plt.scatter(df_lin[feature2].values.ravel() , np.log10(df_lin[target1].values.ravel()/df_lin["38.mGluino"].values.ravel()**2), alpha=0.3)
plt.xlabel(str(feature1+feature2))
plt.ylabel(target1)
fig1.show()

my_mean = (df_lin[feature1].values.ravel() + df_lin[feature2].values.ravel())/2.

fig2 = plt.figure(2)
plt.scatter(my_mean, np.log10(df_lin[target1].values.ravel()/df_lin["38.mGluino"].values.ravel()**2), alpha=0.3)
plt.xlabel('Mean')
plt.ylabel(target1)
plt.show()
"""
plt.show()



