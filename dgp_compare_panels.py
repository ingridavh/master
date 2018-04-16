#Takes the file bigapples.txt and divides points according to size
#Plots histograms of relative error

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.axes as mpx
import numpy as np
import pandas as pd
from scipy.stats import norm
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error

mpl.style.use('ingrid_style')

myfile_bm = open('bm_dLuL_sigmam2/2000t_nomean_CrbfW_noalpha_outliers.dat')
myfile_out = open('bm_dLuL_sigmam2/2000t_nomean_CrbfW_noalpha_nooutliers.dat')
myfile_cut = open('bm_dLuL_sigmam2/2000t_nomean_CrbfW_noalpha_cut16.dat')
myfile_mean = open('bm_dLuL_sigmam2/2000t_mean_CrbfW_noalpha_outliers.dat')
myfile_kernel = open('bm_dLuL_sigmam2/2000t_nomean_matern15_noalpha_outliers.dat')

#myfile_list = [myfile_bm, myfile_out, myfile_cut, myfile_mean, myfile_kernel]
    
################################################
# Try accessing data with pandas               #
################################################

df_lin_bm = pd.read_csv(myfile_bm, sep=" ", header=None, skiprows=1, skipinitialspace=True)
df_lin_out = pd.read_csv(myfile_out, sep=" ", header=None, skiprows=1, skipinitialspace=True)
df_lin_cut = pd.read_csv(myfile_cut, sep=" ", header=None, skiprows=1, skipinitialspace=True)
df_lin_mean = pd.read_csv(myfile_mean, sep=" ", header=None, skiprows=1, skipinitialspace=True)
df_lin_kernel = pd.read_csv(myfile_kernel, sep=" ", header=None, skiprows=1, skipinitialspace=True)

df_lin_list = [df_lin_bm, df_lin_out, df_lin_cut, df_lin_mean, df_lin_kernel]

for df_lin in df_lin_list:
        df_lin.columns = ["Error", "Mus", "Sigmas", "Y_test", "Mg", "Mq", "Unnamed: 4"]
        df_lin = df_lin.drop(["Unnamed: 4"], axis=1)

        
errors_bm = df_lin_bm["Error"].values.ravel()
mus_bm = df_lin_bm["Mus"].values.ravel()
y_test_bm = df_lin_bm["Y_test"].values.ravel()
m1_bm = df_lin_bm["Mg"].values.ravel()
m2_bm = df_lin_bm["Mq"].values.ravel()

errors_out = df_lin_out["Error"].values.ravel()
mus_out = df_lin_out["Mus"].values.ravel()
y_test_out = df_lin_out["Y_test"].values.ravel()
m1_out = df_lin_out["Mg"].values.ravel()
m2_out = df_lin_out["Mq"].values.ravel()

errors_cut = df_lin_cut["Error"].values.ravel()
mus_cut = df_lin_cut["Mus"].values.ravel()
y_test_cut = df_lin_cut["Y_test"].values.ravel()
m1_cut = df_lin_cut["Mg"].values.ravel()
m2_cut = df_lin_cut["Mq"].values.ravel()

errors_mean = df_lin_mean["Error"].values.ravel()
mus_mean = df_lin_mean["Mus"].values.ravel()
y_test_mean = df_lin_mean["Y_test"].values.ravel()
m1_mean = df_lin_mean["Mg"].values.ravel()
m2_mean = df_lin_mean["Mq"].values.ravel()

errors_kernel = df_lin_kernel["Error"].values.ravel()
mus_kernel = df_lin_kernel["Mus"].values.ravel()
y_test_kernel = df_lin_kernel["Y_test"].values.ravel()
m1_kernel = df_lin_kernel["Mg"].values.ravel()
m2_kernel = df_lin_kernel["Mq"].values.ravel()

# To modify histograms to stay inside limits [-1,1]
mod = True
lims = 1

mus_bm = 2*np.log10(m1_bm) + mus_bm
y_test_bm = 2*np.log10(m1_bm) + y_test_bm

mus_out = 2*np.log10(m1_out) + mus_out
y_test_out = 2*np.log10(m1_out) + y_test_out

mus_cut = 2*np.log10(m1_cut) + mus_cut
y_test_cut = 2*np.log10(m1_cut) + y_test_cut

mus_mean = 2*np.log10(m1_mean) + mus_mean
y_test_mean = 2*np.log10(m1_mean) + y_test_mean

mus_kernel = 2*np.log10(m1_kernel) + mus_kernel
y_test_kernel = 2*np.log10(m1_kernel) + y_test_kernel


#############################################
# Organize by NLO-size                      #
#############################################

print "Smallest NLO:", min(y_test_bm)
print "Largest NLO: ", max(y_test_bm)

min_pow = int(min(y_test_bm))
max_pow = int(max(y_test_bm))

num_splits = max_pow - min_pow
print "There are ", num_splits, " powers" 

errors_split_bm = []
errors_split_out = []
errors_split_cut = []
errors_split_mean = []
errors_split_kernel = []

target_test_split_bm = []
target_predicted_split_bm = []

N = len(y_test_bm)

# Rearrange arrays, make a matrix [a][b] where a is exponent, and b are points
for i in range(num_splits+2):
    #BM 
    temp_list_bm = []
    temp_list_out = []
    temp_list_cut = []
    temp_list_mean = []
    temp_list_kernel = []
    
    temp_list_targets = []
    temp_list_predicted_bm = []
    
    for j in range(N):
        if (y_test_bm[j] >= (i-(num_splits+1)+max_pow)) and (y_test_bm[j] <= (i-(num_splits)+max_pow)):
            temp_list_bm.append(errors_bm[j])
            temp_list_out.append(errors_out[j])
            temp_list_cut.append(errors_cut[j])
            temp_list_mean.append(errors_mean[j])
            temp_list_kernel.append(errors_kernel[j])
            
            temp_list_targets.append(y_test_bm[j])
            
    errors_split_bm.append(np.asarray(temp_list_bm))
    errors_split_out.append(np.asarray(temp_list_out))
    errors_split_cut.append(np.asarray(temp_list_cut))
    errors_split_mean.append(np.asarray(temp_list_mean))
    errors_split_kernel.append(np.asarray(temp_list_kernel))
    
    target_test_split_bm.append(np.array(temp_list_targets))


    
#Turn back into arrays
errors_split_bm = np.asarray(errors_split_bm)
errors_split_out = np.asarray(errors_split_out)
errors_split_cut = np.asarray(errors_split_cut)
errors_split_mean = np.asarray(errors_split_mean)
errors_split_kernel = np.asarray(errors_split_kernel)

target_test_split = np.asarray(target_test_split_bm)

###########################################
# Plot splitted dataset                   #
###########################################

mu_list_bm = []
sigma_list_bm = []

mu_list_out = []
sigma_list_out = []

mu_list_cut = []
sigma_list_cut = []

mu_list_mean = []
sigma_list_mean = []

mu_list_kernel = []
sigma_list_kernel = []

for i in range(num_splits+2):
    # BM
    if len(errors_split_bm[i]) >= 1:
            
        # Find mu and sigma for Gaussian fit
        (mu_bm, sigma_bm) = norm.fit(errors_split_bm[i])
        mu_list_bm.append(mu_bm); sigma_list_bm.append(sigma_bm)

    else:
        mu_bm = 0; sigma_bm = 0
        mu_list_bm.append(mu_bm); sigma_list_bm.append(sigma_bm)

    # Outliers
    if len(errors_split_out[i]) >= 1:
            
        # Find mu and sigma for Gaussian fit
        (mu_out, sigma_out) = norm.fit(errors_split_out[i])
        mu_list_out.append(mu_out); sigma_list_out.append(sigma_out)

    else:
        mu_out = 0; sigma_out = 0
        mu_list_out.append(mu_out); sigma_list_out.append(sigma_out)

    # Cut at -16
    if len(errors_split_cut[i]) >= 1:
            
        # Find mu and sigma for Gaussian fit
        (mu_cut, sigma_cut) = norm.fit(errors_split_cut[i])
        mu_list_cut.append(mu_cut); sigma_list_cut.append(sigma_cut)

    else:
        mu_cut = 0; sigma_cut = 0
        mu_list_cut.append(mu_cut); sigma_list_cut.append(sigma_cut)

    # Mean
    if len(errors_split_mean[i]) >= 1:
            
        # Find mu and sigma for Gaussian fit
        (mu_mean, sigma_mean) = norm.fit(errors_split_mean[i])
        mu_list_mean.append(mu_mean); sigma_list_mean.append(sigma_mean)

    else:
        mu_mean = 0; sigma_mean = 0
        mu_list_mean.append(mu_mean); sigma_list_mean.append(sigma_mean)

    # Kernel
    if len(errors_split_kernel[i]) >= 1:
            
        # Find mu and sigma for Gaussian fit
        (mu_kernel, sigma_kernel) = norm.fit(errors_split_kernel[i])
        mu_list_kernel.append(mu_kernel); sigma_list_kernel.append(sigma_kernel)

    else:
        mu_kernel = 0; sigma_kernel = 0
        mu_list_kernel.append(mu_kernel); sigma_list_kernel.append(sigma_kernel)



        
mu_array_bm = np.asarray(mu_list_bm)
sigma_array_bm = np.asarray(sigma_list_bm)

mu_array_out = np.asarray(mu_list_out)
sigma_array_out = np.asarray(sigma_list_out)

mu_array_cut = np.asarray(mu_list_cut)
sigma_array_cut = np.asarray(sigma_list_cut)

mu_array_mean = np.asarray(mu_list_mean)
sigma_array_mean = np.asarray(sigma_list_mean)

mu_array_kernel = np.asarray(mu_list_kernel)
sigma_array_kernel = np.asarray(sigma_list_kernel)


#######################################################################
# Plot mus with errorbars                                             #
#######################################################################


x = np.linspace(min_pow,max_pow+1,num_splits+2)
x_corr = x + 0.25
realx = np.zeros(10)-3.1
realy = np.linspace(-1,1, 10)

cut_int = 0
xmin = -12
xmax = 7.5
ylims = 1

yerr_bm = sigma_array_bm[cut_int:]
yerr_out = sigma_array_out[cut_int:]
yerr_cut = sigma_array_cut[cut_int:]
yerr_mean = sigma_array_mean[cut_int:]
yerr_kernel = sigma_array_kernel[cut_int:]

#yerr_ = np.zeros(len(x[cut_int:]))+0.4

#colors = ['red', 'mediumseagreen', 'crimson','mediumvioletred', 'magenta', 'darkviolet']
colors = ['red', 'orange', 'dodgerblue', 'slateblue', 'blue', 'steelblue']

fig = plt.figure()
ax1 = fig.add_subplot(221)
plt.text(5.5,0.8, 'a)', fontsize=20)
plt.plot(realx, realy, '--', label='0.02 event limit', color=colors[0])
plt.errorbar(x[cut_int:], mu_array_bm[cut_int:], yerr=yerr_bm, fmt='o', linewidth=2.0, label='BM', color=colors[1])
plt.errorbar(x_corr[cut_int:], mu_array_out[cut_int:], yerr=yerr_out, fmt='o', linewidth=2.0, label='No outliers', color=colors[2])
plt.legend(loc='lower right', fontsize='x-large')
plt.ylabel(r'$\bar{\varepsilon}$', fontsize='xx-large')
plt.xlabel(r'$\log_{10} \sigma/\sigma^0$, $\sigma^0 = 1$fb', size='xx-large')
plt.ylim([-ylims,ylims])
plt.xlim([xmin, xmax])
plt.xticks(np.arange(xmin, xmax, 2.0))

ax2 = fig.add_subplot(222)
plt.text(5.5,0.8, 'b)', fontsize=20)
plt.plot(realx, realy, '--', color=colors[0])
plt.errorbar(x[cut_int:], mu_array_bm[cut_int:], yerr=yerr_bm, fmt='o', linewidth=2.0, label='BM', color=colors[1])
plt.errorbar(x_corr[cut_int:], mu_array_cut[cut_int:], yerr=yerr_cut, fmt='o', linewidth=2.0, label= r'$\sigma > 10^{-16}$', color=colors[3])
plt.legend(loc='lower right', fontsize='x-large')
plt.ylabel(r'$\bar{\varepsilon}$', fontsize='xx-large')
plt.xlabel(r'$\log_{10} \sigma/\sigma^0$, $\sigma^0 = 1$fb', size='xx-large')
plt.ylim([-ylims,ylims])
plt.xlim([xmin, xmax])
plt.xticks(np.arange(xmin, xmax, 2.0))

ax3 = fig.add_subplot(223)
plt.text(5.5,0.8, 'c)', fontsize=20)
plt.plot(realx, realy, '--', color=colors[0])
plt.errorbar(x[cut_int:], mu_array_bm[cut_int:], yerr=yerr_bm, fmt='o', linewidth=2.0, label='BM', color=colors[1])
plt.errorbar(x_corr[cut_int:], mu_array_mean[cut_int:], yerr=yerr_mean, fmt='o', linewidth=2.0, label=r'$\bar{m}$', color=colors[4])
plt.legend(loc='lower right', fontsize='x-large')
plt.ylabel(r'$\bar{\varepsilon}$', fontsize='xx-large')
plt.xlabel(r'$\log_{10} \sigma/\sigma^0$, $\sigma^0 = 1$fb', size='xx-large')
plt.ylim([-ylims,ylims])
plt.xlim([xmin, xmax])
plt.xticks(np.arange(xmin, xmax, 2.0))


ax4 = fig.add_subplot(224)
plt.text(5.5,0.8, 'd)', fontsize=20)
plt.plot(realx, realy, '--', color=colors[0])
plt.errorbar(x[cut_int:], mu_array_bm[cut_int:], yerr=yerr_bm, fmt='o', linewidth=2.0, label='BM', color=colors[1])
plt.errorbar(x_corr[cut_int:], mu_array_kernel[cut_int:], yerr=yerr_kernel, fmt='o', linewidth=2.0, label='Matern', color=colors[5])
plt.legend(loc='lower right', fontsize='x-large')
plt.ylabel(r'$\bar{\varepsilon}$', fontsize='xx-large')
plt.xlabel(r'$\log_{10} \sigma/\sigma^0$, $\sigma^0 = 1$fb', size='xx-large')
plt.ylim([-ylims,ylims])
plt.xlim([xmin, xmax])
plt.xticks(np.arange(xmin, xmax, 2.0))

plt.show()

#plt.title(r'$\sigma \geq$ 1e-16 for $\tilde{d}_L\tilde{u}_L$')
#plt.savefig('/home/ingrid/Documents/Master/Programs/DistributedGaussianProcesses/bm_plots/dLdL_compare_panels_nu15.pdf')


