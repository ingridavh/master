import numpy as np
import sklearn as sk
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from sklearn.externals import joblib

mpl.style.use('ingrid_style')

# Read from file
myfile = open('dRdR_2000x10_prospino.dat')
df_lin = pd.read_csv(myfile, sep=" ", header=None, skiprows=1, skipinitialspace=True)

#df_lin.columns = ["Error", "Mus", "Sigmas", "Y_test", "Mg", "Mq", "Unnamed: 4"]
# Abel
df_lin.columns = ["Index", "Mg", "Mq", "Mus", "Sigmas"]

#df_lin = df_lin.drop(["Unnamed: 4"], axis=1)

mus = df_lin["Mus"].values.ravel()
sigmas = df_lin["Sigmas"].values.ravel()
mg = df_lin["Mg"].values.ravel()
mq = df_lin["Mq"].values.ravel()



mus += 2*np.log10(mg)

# From Prospino

df_dldl = pd.read_csv('prospino_dl_ss.dat', sep=" ", header=None, skiprows=0, skipinitialspace=True)
df_dldl.columns = ["1.mdL","2.mg","3.m_mean","4.LO","5.NLO", "6.Unnamed"]

mdL = df_dldl["1.mdL"].values.ravel()

################################################################
# W/ error bars from Prospino

df_dldl_err = pd.read_csv('prospino_dldl_rel_err.dat', sep=" ", header=None, skiprows=0, skipinitialspace=True)
df_dldl_err.columns = ["Process", "i1", "i2", "dummy0", "dummy1", "scafac", "m1", "m2", "angle", "LO[pb]", "rel_error", "NLO[pb]", "rel_error", "K", "LO_ms[pb]", "NLO_ms[pb]", "Unnamed"]

################################################################

# dLdL
NLO_dldl_err = df_dldl_err["NLO_ms[pb]"].values.ravel()*1e03
N = len(NLO_dldl_err)
NLO_1_dldl = np.log10(NLO_dldl_err[0:N-2:3])
NLO_2_dldl = np.log10(NLO_dldl_err[1:N-1:3])
NLO_3_dldl = np.log10(NLO_dldl_err[2:N:3])


g = plt.figure(3)
plt.plot(mq, mus, alpha= 0.7, color='lightseagreen', zorder=2)
plt.errorbar(mdL, NLO_2_dldl, yerr=[5*(NLO_2_dldl-NLO_3_dldl), 5*(NLO_1_dldl-NLO_2_dldl)], fmt='+', color='navy', zorder=3)
plt.xlabel(r'$\bar{m}_{\tilde{d}_L}$')
plt.ylabel(r'$\log_{10}(\sigma/\sigma^0)$, $\sigma^0 = 1$ fb')
plt.legend([r'Gaussian Processes $d_Ld_L$','Prospino $d_Ld_L$'], fontsize='x-large')
plt.fill_between(mq, mus - 5*sigmas, mus + 5*sigmas, alpha=0.3, color='lightseagreen', zorder=1)
g.show()

plt.show()


