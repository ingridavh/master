import numpy as np
import sklearn as sk
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from sklearn.externals import joblib

mpl.style.use('ingrid_style')

# Read from file
myfile = open('uRdR_3000x10_prospino.dat')

df_lin = pd.read_csv(myfile, sep=" ", header=None, skiprows=1, skipinitialspace=True)
#df_lin.columns = ["Error", "Mus", "Sigmas", "Y_test", "Mg", "Mq", "Unnamed: 4"]
#df_lin = df_lin.drop(["Unnamed: 4"], axis=1)

# Abel
df_lin.columns = ["Index","Md", "Mg", "Mu", "Mus", "Sigmas"]

mus = df_lin["Mus"].values.ravel()
sigmas = df_lin["Sigmas"].values.ravel()
mg = df_lin["Mg"].values.ravel()
mq = df_lin["Md"].values.ravel()

mus += 2*np.log10(mg)

# From Prospino

df_dldl = pd.read_csv('prospino_dl_ss.dat', sep=" ", header=None, skiprows=0, skipinitialspace=True)
df_dldl.columns = ["1.mdL","2.mg","3.m_mean","4.LO","5.NLO", "6.Unnamed"]

mdL = df_dldl["1.mdL"].values.ravel()

################################################################
# W/ error bars from Prospino

df_dlul_err = pd.read_csv('prospino_dlul_rel_err.dat', sep=" ", header=None, skiprows=0, skipinitialspace=True)
df_dlul_err.columns = ["Process", "i1", "i2", "dummy0", "dummy1", "scafac", "m1", "m2", "angle", "LO[pb]", "rel_error", "NLO[pb]", "rel_error", "K", "LO_ms[pb]", "NLO_ms[pb]", "Unnamed"]

################################################################

# dLuL
NLO_dlul_err = df_dlul_err["NLO_ms[pb]"].values.ravel()*1e03
N = len(NLO_dlul_err)
NLO_1_dlul = np.log10(NLO_dlul_err[0:N-2:3])
NLO_2_dlul = np.log10(NLO_dlul_err[1:N-1:3])
NLO_3_dlul = np.log10(NLO_dlul_err[2:N:3])

scaling = 2

g = plt.figure(3)
plt.plot(mq, mus, alpha= 0.7, color='lightseagreen', zorder=2)
plt.errorbar(mdL, NLO_2_dlul, yerr=[2*(NLO_2_dlul-NLO_3_dlul), 2*(NLO_1_dlul-NLO_2_dlul)], fmt='+', color='navy', zorder=3)
plt.xlabel(r'$\bar{m}_{\tilde{d}_L}$')
plt.ylabel(r'$\log_{10}(\sigma/\sigma^0)$, $\sigma^0 = 1$ fb')
plt.legend([r'Gaussian Processes $d_Ld_L$','Prospino $d_Ld_L$'], fontsize='x-large')
plt.fill_between(mq, mus - scaling*sigmas, mus + scaling*sigmas, alpha=0.3, color='lightseagreen', zorder=1)
plt.ylim([-6,3])
g.show()

plt.show()


