import numpy as np
import sklearn as sk
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from sklearn.externals import joblib

mpl.style.use('ingrid_style')

# Read from file
myfile_dRdR = open('dRdR_2000x10_prospino.dat')
df_lin_dRdR = pd.read_csv(myfile_dRdR, sep=" ", header=None, skiprows=1, skipinitialspace=True)

myfile_uRdR = open('uRdR_2000x10_prospino.dat')
df_lin_uRdR = pd.read_csv(myfile_uRdR, sep=" ", header=None, skiprows=1, skipinitialspace=True)

# Abel dRdR
df_lin_dRdR.columns = ["Index", "Mg", "Mq", "Mus", "Sigmas"]
df_lin_uRdR.columns = ["Index","Md", "Mg", "Mu", "Mus", "Sigmas"]

mus_dRdR = df_lin_dRdR["Mus"].values.ravel()
sigmas_dRdR = df_lin_dRdR["Sigmas"].values.ravel()
mg_dRdR = df_lin_dRdR["Mg"].values.ravel()
mq_dRdR = df_lin_dRdR["Mq"].values.ravel()

mus_dRdR += 2*np.log10(mg_dRdR)

# Abel uRdR

mus_uRdR = df_lin_uRdR["Mus"].values.ravel()
sigmas_uRdR = df_lin_uRdR["Sigmas"].values.ravel()
mg_uRdR = df_lin_uRdR["Mg"].values.ravel()
mq_uRdR = df_lin_uRdR["Md"].values.ravel()

mus_uRdR += 2*np.log10(mg_uRdR)

# From Prospino

df_dldl = pd.read_csv('prospino_dl_ss.dat', sep=" ", header=None, skiprows=0, skipinitialspace=True)
df_dldl.columns = ["1.mdL","2.mg","3.m_mean","4.LO","5.NLO", "6.Unnamed"]

mdL = df_dldl["1.mdL"].values.ravel()

################################################################
# W/ error bars from Prospino

df_dldl_err = pd.read_csv('prospino_dldl_rel_err.dat', sep=" ", header=None, skiprows=0, skipinitialspace=True)
df_dldl_err.columns = ["Process", "i1", "i2", "dummy0", "dummy1", "scafac", "m1", "m2", "angle", "LO[pb]", "rel_error", "NLO[pb]", "rel_error", "K", "LO_ms[pb]", "NLO_ms[pb]", "Unnamed"]


df_dlul_err = pd.read_csv('prospino_dlul_rel_err.dat', sep=" ", header=None, skiprows=0, skipinitialspace=True)
df_dlul_err.columns = ["Process", "i1", "i2", "dummy0", "dummy1", "scafac", "m1", "m2", "angle", "LO[pb]", "rel_error", "NLO[pb]", "rel_error", "K", "LO_ms[pb]", "NLO_ms[pb]", "Unnamed"]

################################################################

# dLdL
NLO_dldl_err = df_dldl_err["NLO_ms[pb]"].values.ravel()*1e03
N = len(NLO_dldl_err)
NLO_1_dldl = np.log10(NLO_dldl_err[0:N-2:3])
NLO_2_dldl = np.log10(NLO_dldl_err[1:N-1:3])
NLO_3_dldl = np.log10(NLO_dldl_err[2:N:3])

# dLuL
NLO_dlul_err = df_dlul_err["NLO_ms[pb]"].values.ravel()*1e03
N = len(NLO_dlul_err)
NLO_1_dlul = np.log10(NLO_dlul_err[0:N-2:3])
NLO_2_dlul = np.log10(NLO_dlul_err[1:N-1:3])
NLO_3_dlul = np.log10(NLO_dlul_err[2:N:3])

scaling = 50
scaling2 = 5

g = plt.figure(3)
plt.plot(mq_dRdR, mus_dRdR, alpha= 0.7, color='lightseagreen', zorder=2, label=r'Gaussian Processes $\tilde{d}_L\tilde{d}_L$, %.f $\sigma$' % scaling)
plt.fill_between(mq_dRdR, mus_dRdR - scaling*sigmas_dRdR, mus_dRdR + scaling*sigmas_dRdR, alpha=0.3, color='lightseagreen', zorder=1)
plt.plot(mq_uRdR, mus_uRdR, alpha= 0.7, color='orangered', zorder=2, label=r'Gaussian Processes $\tilde{u}_L\tilde{d}_L$, %.f $\sigma$' % scaling)
plt.fill_between(mq_uRdR, mus_uRdR - scaling*sigmas_uRdR, mus_uRdR + scaling*sigmas_uRdR, alpha=0.3, color='orangered', zorder=1)

# Prospino dLdL
plt.errorbar(mdL, NLO_2_dldl, yerr=[scaling2*(NLO_2_dldl-NLO_3_dldl), scaling2*(NLO_1_dldl-NLO_2_dldl)], fmt='+', color='navy', zorder=3, label=r'Prospino $\tilde{d}_L\tilde{d}_L$, %.f $\sigma$' % scaling2)
# Prospino uLdL
plt.errorbar(mdL, NLO_2_dlul, yerr=[scaling2*(NLO_2_dlul-NLO_3_dlul), scaling2*(NLO_1_dlul-NLO_2_dlul)], fmt='+', color='green', zorder=3, label=r'Prospino $\tilde{u}_L\tilde{d}_L$, %.f $\sigma$' % scaling2)

plt.xlabel(r'$\bar{m}_{\tilde{d}_L}$')
plt.ylabel(r'$\log_{10}(\sigma/\sigma_0)$, $\sigma_0 = 1$ fb')
plt.legend(fontsize='x-large')

g.show()

plt.show()
