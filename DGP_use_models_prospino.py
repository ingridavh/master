import numpy as np
import sklearn as sk
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from sklearn.externals import joblib

mpl.style.use('ingrid_style')

#squark_names = ['uL', 'dL', 'sL', 'cL', 'uR', 'dR', 'sR', 'cR']
squark_names = ['uR', 'dR']
training_points = '_2000_matern'

# Feature values
N = 100
mq_min = 200
mq_max = 2400
m_rest = 1000

mg_test = np.zeros(N)+500

mq_test = np.linspace(mq_min, mq_max, N)
mean_test = (m_rest*7+mq_test)/8.
mq_test_2 = np.zeros(N)+1000

features_test = np.zeros((N, 3))
features_test_2 = np.zeros((N, 4))

features_test[:,0] = mg_test
features_test[:,1] = mq_test
features_test[:,2] = mean_test

features_test_2[:,0] = mg_test
features_test_2[:,2] = mq_test
features_test_2[:,1] = mq_test_2
features_test_2[:,3] = mean_test


mu_array = np.zeros((36, N))
sigma_array = np.zeros((36, N))
mus_total = np.zeros(N)
sigmas_total = np.zeros(N)

cs_num = np.linspace(0, 35, 36)
cs_sum = 0

for i in range(len(squark_names)):
    for j in range(i, len(squark_names)):
        squark1 = squark_names[i]
        squark2 = squark_names[j]

        squark_combo = squark1+squark2
        print squark_combo, cs_sum
        
        # Load trained model
        squark_job = joblib.load(squark_combo+training_points)

        # Predict using model
        if i == j:
            my_features_test = features_test
        else:
            my_features_test = features_test_2


        mu_arrays, sigma_arrays = squark_job.predict(my_features_test, return_std = True)
        mu_arrays +=  2*np.log10(mg_test)

        mu_array[cs_sum] = mu_arrays
        sigma_array[cs_sum] = sigma_arrays

        mus_total += 10**mu_arrays
        sigmas_total += sigma_arrays**2

        cs_sum += 1

mus_total = np.log10(mus_total)
sigmas_total = np.sqrt(sigmas_total)


mu_dldl = mu_array[2]
mu_dlul = mu_array[1]
sigma_dldl = sigma_array[2]
sigma_dlul = sigma_array[1]

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

"""
# dLdL lines
h = plt.figure(1)
plt.plot(mq_test, mu_dldl, alpha= 0.6, color='lightseagreen', zorder=4)
plt.plot(mdL, NLO_2_dldl, alpha= 0.6, color='navy', zorder=3)
plt.xlabel(r'$\bar{m}_{\tilde{d}_L}$')
plt.ylabel(r'$\log_{10}(\sigma/\sigma^0)$, $\sigma^0 = 1$ fb')
plt.fill_between(mq_test, mu_dldl - sigma_dldl, mu_dldl + sigma_dldl, alpha=0.3, color='lightseagreen', zorder=1)
plt.fill_between(mdL, NLO_3_dldl, NLO_1_dldl, alpha=0.3, color='navy', zorder=2)
plt.legend([r'Gaussian Processes $d_Ld_L$', r'Prospino $d_Ld_L$'], fontsize='x-large')
plt.title('Option 1')
#plt.savefig('plots/prospino_dLdL_5000.pdf')
h.show()

# dLuL lines
m = plt.figure(2)
plt.plot(mq_test, mu_dlul, alpha=0.7, color='lightseagreen', zorder=4)
plt.plot(mdL, NLO_2_dlul, alpha=0.7, color='navy', zorder=3)
plt.fill_between(mq_test, mu_dlul - sigma_dlul, mu_dlul + sigma_dlul, alpha=0.3, color='lightseagreen', zorder=1)
plt.fill_between(mdL, NLO_3_dlul, NLO_1_dlul, alpha=0.3, color='navy', zorder=2)
plt.legend([r'Gaussian Processes $d_Lu_L$','Prospino $d_Lu_L$'], fontsize='x-large')
plt.xlabel(r'$\bar{m}_{\tilde{d}_L}$')
plt.ylabel(r'$\log_{10}(\sigma/\sigma^0)$, $\sigma^0 = 1$ fb')
plt.title('Option 1')
#plt.savefig('plots/prospino_dLdL_5000.pdf')
h.show()
"""


# dLdL errors
g = plt.figure(3)
plt.plot(mq_test, mu_dldl, alpha= 0.7, color='lightseagreen', zorder=2)
plt.errorbar(mdL, NLO_2_dldl, yerr=[NLO_2_dldl-NLO_3_dldl, NLO_1_dldl-NLO_2_dldl], fmt='+', color='navy', zorder=3)
plt.xlabel(r'$\bar{m}_{\tilde{d}_L}$')
plt.ylabel(r'$\log_{10}(\sigma/\sigma^0)$, $\sigma^0 = 1$ fb')
plt.legend([r'Gaussian Processes $d_Ld_L$','Prospino $d_Ld_L$'], fontsize='x-large')
plt.fill_between(mq_test, mu_dldl - sigma_dldl, mu_dldl + sigma_dldl, alpha=0.3, color='lightseagreen', zorder=1)
plt.title('Option 2')
g.show()

# dLuL errors
n = plt.figure(4)
plt.plot(mq_test, mu_dlul, alpha=0.7, color='lightseagreen', zorder=2)
plt.errorbar(mdL, NLO_2_dlul, yerr=[NLO_2_dlul-NLO_3_dlul, NLO_1_dlul-NLO_2_dlul], fmt='+', color='navy', zorder=3)
plt.xlabel(r'$\bar{m}_{\tilde{d}_L}$')
plt.ylabel(r'$\log_{10}(\sigma/\sigma^0)$, $\sigma^0 = 1$ fb')
plt.legend([r'Gaussian Processes $d_Lu_L$','Prospino $d_Lu_L$'], fontsize='x-large')
plt.fill_between(mq_test, mu_dlul - sigma_dlul, mu_dlul + sigma_dlul, alpha=0.3, color='lightseagreen', zorder=1)
plt.title('Option 2')
g.show()

# dLdL and dLuL errors
myfig = plt.figure(5)
plt.plot(mq_test, mu_dldl, alpha= 0.7, color='lightseagreen', zorder=4)
plt.plot(mq_test, mu_dlul, alpha=0.7, color='orangered', zorder=3)
plt.errorbar(mdL, NLO_2_dldl, yerr=[2*(NLO_2_dldl-NLO_3_dldl), 2*(NLO_1_dldl-NLO_2_dldl)], fmt='+', color='navy', zorder=6, markersize=5)
plt.errorbar(mdL, NLO_2_dlul, yerr=[2*(NLO_2_dlul-NLO_3_dlul), 2*(NLO_1_dlul-NLO_2_dlul)], fmt='+', color='green', zorder=5, markersize=5)
plt.xlabel(r'$\bar{m}_{\tilde{d}_L}$')
plt.ylabel(r'$\log_{10}(\sigma/\sigma^0)$, $\sigma^0 = 1$ fb')
plt.legend([r'Gaussian Processes $d_Ld_L$', r'Gaussian Processes $d_Lu_L$','Prospino $d_Ld_L$', 'Prospino $d_Lu_L$'], fontsize='x-large')
plt.fill_between(mq_test, mu_dldl - 2*sigma_dldl, mu_dldl + 2*sigma_dldl, alpha=0.3, color='lightseagreen', zorder=2)
plt.fill_between(mq_test, mu_dlul - 2*sigma_dlul, mu_dlul + 2*sigma_dlul, alpha=0.3, color='orangered', zorder=1)
plt.savefig('dldl_ulul_prospino_5000.pdf')
myfig.show()



plt.show()


