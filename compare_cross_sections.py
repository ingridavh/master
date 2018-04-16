import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

mpl.style.use('ingrid_style')

data_lin = pd.read_csv('/home/ingrid/Documents/Master/ML/Abel_all_crossections/data_40k.dat', sep=' ', skipinitialspace=True)

#Create data frames in pandas
df_lin = pd.DataFrame(data_lin)

#mask = df_lin['17.dLdL_NLO'] != 0
#df_lin = df_lin[mask]

#Include outliers, but set to -32
eps = 1e-32
df_lin = df_lin.replace(0.0, eps)

mean_list = ["39.mcL", "40.mdL", "41.mdR", "42.muL", "43.msL", "44.muR", "45.msR", "46.mcR"]
feature_list = ["38.mGluino", "40.mdL"]

plt.scatter(df_lin["42.muL"].values.ravel(), df_lin["39.mcL"].values.ravel())
plt.xlabel('muL')
plt.ylabel('mcL')
plt.show()

fig1 = plt.figure(0)
plt.scatter(df_lin["42.muL"].values.ravel() , np.log10(df_lin["5.cLuL_NLO"].values.ravel()/df_lin["38.mGluino"].values.ravel()**2), alpha=0.3)
plt.scatter(df_lin["39.mcL"].values.ravel() , np.log10(df_lin["5.cLuL_NLO"].values.ravel()/df_lin["38.mGluino"].values.ravel()**2), alpha=0.3)
plt.xlabel('muL')
plt.ylabel('sigma')
fig1.show()

my_mean = (df_lin["42.muL"].values.ravel() + df_lin["39.mcL"].values.ravel())/2

plt.scatter(my_mean, np.log10(df_lin["5.cLuL_NLO"].values.ravel()/df_lin["38.mGluino"].values.ravel()**2), alpha=0.3)
plt.show()

"""
means = df_lin[mean_list].mean(axis=1).values.ravel()

n_features = len(feature_list)
target_list = ["17.dLdL_NLO"]
features = df_lin[feature_list].values
target = df_lin[target_list].values.ravel()

long_features = np.zeros(( len(features), 3 ))
long_features[:,0] = features[:,0]
long_features[:,1] = features[:,1]
long_features[:,2] = means[:,]

target_mg = target/features[:,0]**2
target_mq = target/features[:,1]**2
target_mqmg = target/(features[:,0]**2*features[:,1]**2)
target_fac =  target/features[:,0]**2*(features[:,0]**2+features[:,1]**2)**2

target = np.log10(target)
target_mg = np.log10(target_mg)
target_mq = np.log10(target_mq)
target_fac = np.log10(target_fac)
"""
# sigma
"""
ax1 = plt.subplot(221)
plt.scatter(long_features[:,0], target, alpha=0.5)
plt.ylabel(r'$\log_{10}(\sigma/\sigma^0)$')
plt.setp(ax1.get_xticklabels(), visible=False)
plt.ylim([-32,6])

ax4 = plt.subplot(222, sharey=ax1)
plt.scatter(long_features[:,1], target, alpha=0.5, color='seagreen')
plt.setp(ax4.get_xticklabels(), visible=False)
plt.setp(ax4.get_yticklabels(), visible=False)
plt.ylim([-32,6])

#sigma_mg
ax3 = plt.subplot(223, sharex=ax1)
plt.scatter(long_features[:,0], target_mg, alpha=0.5)
plt.ylabel(r'$\log_{10}(\sigma_{m_{\tilde{g}}}/\sigma^0m^0)$')
plt.xlabel(r'm_{\tilde{g}}')
plt.ylim([-32,6])

ax6 = plt.subplot(224, sharex=ax4, sharey=ax3)
plt.scatter(long_features[:,1], target_mg, alpha=0.5, color='seagreen')
plt.setp(ax6.get_yticklabels(), visible=False)
plt.xlabel(r'm_{\tilde{q}}')
plt.ylim([-32,6])
"""
#plt.savefig('/home/ingrid/Documents/Master/uiofysmaster-master/dingsen/figures_evaluating_cross_sections/data_transformation_sigma_sigmam2.pdf')
"""
q = plt.figure(5)
plt.scatter(long_features[:,1], target, alpha=0.5, color='seagreen')
plt.ylabel(r'$\log_{10}(\sigma/\sigma^0)$')
plt.xlabel('m_{\tilde{q}}')
plt.savefig('sigma.pdf')
q.show()

m = plt.figure(4)
plt.scatter(long_features[:,0], target_mg, alpha=0.5)
plt.ylabel(r'$\log_{10}(\sigma_{m_{\tilde{g}}}/\sigma^0m^0)$')
plt.xlabel(r'm_{\tilde{g}}')
m.show()

n = plt.figure(3)
plt.scatter(long_features[:,1], target_mg, alpha=0.5, color='seagreen')
plt.ylabel(r'$\log_{10}(\sigma_{m_{\tilde{g}}}/\sigma^0m^0)$')
plt.xlabel(r'm_{\tilde{q}}')
plt.savefig('sigma_mg.pdf')
n.show()

g = plt.figure(1)
plt.scatter(features[:,0], target_fac, color='seagreen')
plt.ylabel(r'$\log_{10}(\sigma_{fac}m^0/\sigma^0)$')
plt.xlabel(r'm_{\tilde{g}}')
g.show()

f = plt.figure(2)
plt.scatter(features[:,1], target_fac, color='seagreen')
plt.ylabel(r'$\log_{10}(\sigma_{fac}m^0/\sigma^0)$')
plt.xlabel(r'm_{\tilde{q}}')
plt.savefig('sigma_fac.pdf')
f.show()

plt.show()
"""

#print max(target_mg), min(target_mg)

"""
plt.scatter(features[:,1], target_mg, alpha=0.5, color='seagreen')
plt.ylabel(r'$\log_{10}(\sigma_{m_{\tilde{g}}}/\sigma^0m^0)$')
plt.xlabel(r'm_{\tilde{q}}')
#plt.savefig('sigma_w_outliers.pdf')
plt.show()
"""
