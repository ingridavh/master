#Takes the file bigapples.txt and divides points according to size
#Plots histograms of relative error

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.axes as mpx
import numpy as np
import pandas as pd
from scipy.stats import norm
import sys

mpl.style.use('ingrid_style')

if len(sys.argv) >= 2:
        myfile = open(sys.argv[1])
else:
        print "Error! No input file was provided."
        sys.exit(1)

if len(sys.argv) >= 3:
    my_kernel = str(sys.argv[2])

if len(sys.argv) >= 4:
        myarg = str(sys.argv[3])
else:
        myarg = False
    
################################################
# Try accessing data with pandas               #
################################################

df_lin = pd.read_csv(myfile, sep=" ", header=None, skiprows=1, skipinitialspace=True)
df_lin.columns = ["Error", "Mus", "Sigmas", "Y_test", "Mg", "Mq", "Unnamed: 4"]
df_lin = df_lin.drop(["Unnamed: 4"], axis=1)

errors = df_lin["Error"].values.ravel()
mus = df_lin["Mus"].values.ravel()
y_test = df_lin["Y_test"].values.ravel()
m1 = df_lin["Mg"].values.ravel()
m2 = df_lin["Mq"].values.ravel()
        
# To modify histograms to stay inside limits [-1,1]
mod = True
lims = 1

if myarg == 'mg': # Change back to sigma^B
        mus = 2*np.log10(m1) + mus
        y_test = 2*np.log10(m1) + y_test
elif myarg == 'mq':
        mus = mus - 2*np.log10(m2)
        y_test = y_test - 2*np.log10(m2)
elif myarg == 'mr':
        mus = mus + 2*np.log10(m1) - 4*np.log10(m2)
        y_test = y_test + 2*np.log10(m1) - 4*np.log10(m2)
elif myarg == 'mgpos':
        mus = mus - 2*np.log10(m1)
        y_test = y_test - 2*np.log10(m1)
        
#############################################
# Organize by NLO-size                      #
#############################################

print "Smallest NLO:", min(y_test)
print "Largest NLO: ", max(y_test)

min_pow = int(min(y_test))
max_pow = int(max(y_test))

num_splits = max_pow - min_pow
print "There are ", num_splits, " powers" 

errors_split = []
target_test_split = []

N = len(y_test)

# Rearrange arrays, make a matrix [a][b] where a is exponent, and b are points
for i in range(num_splits+2):
    temp_list = []
    temp_list_targets = []
    for j in range(N):
        if (y_test[j] >= (i-(num_splits+1)+max_pow)) and (y_test[j] <= (i-(num_splits)+max_pow)):
            temp_list.append(errors[j])
            temp_list_targets.append(y_test[j])
    errors_split.append(np.asarray(temp_list))
    target_test_split.append(np.array(temp_list_targets))

#Turn back into arrays
errors_split = np.asarray(errors_split);
target_test_split = np.asarray(target_test_split)

###########################################
# Plot splitted dataset                   #
###########################################

mu_list = []
sigma_list = []
mean_list = []
std_list = []

for i in range(num_splits+2):
    if len(errors_split[i]) >= 1:
            
        # Find mu and sigma for Gaussian fit
            
        (mu, sigma) = norm.fit(errors_split[i])
        mu_list.append(mu)
        sigma_list.append(sigma)

        # Find mean and std
        
        mean_list.append(np.mean(errors_split[i]))
        std_list.append(np.std(errors_split[i]))
    else:
        mu = 0
        sigma = 0
        mu_list.append(mu); sigma_list.append(sigma)

mu_array = np.asarray(mu_list)
sigma_array = np.asarray(sigma_list)
mean_array = np.asarray(mean_list)
std_array = np.asarray(std_list)

#Print table of values
#print "%-24s %-10s %-10s %-10s" % ("Power", "Mu", "Sigma", "Number of points")
#for i in range(num_splits+2):
#        print " %-4i to %-10i  %-10.4f %10.4f %10.4f" % ((i-num_splits+max_pow-1), (i-num_splits+max_pow), mu_array[i], sigma_array[i], len(errors_split[i]))

#######################################################################
# Plot mus with errorbars                                             #
#######################################################################


x = np.linspace(min_pow,max_pow+1,num_splits+2)
realx = np.zeros(10)-7.1
realy = np.linspace(-1,1, 10)

cut_int = 0
yerr = sigma_array[cut_int:]
yerr_ = np.zeros(len(x[cut_int:]))+0.4
plt.plot(realx, realy, '--')
plt.errorbar(x[cut_int:], mu_array[cut_int:], yerr=yerr, fmt='o', linewidth=2.0)
#plt.title('Mean error with std', size='xx-large')
plt.legend(['1 event limit'], loc='upper left', fontsize='x-large')
plt.ylabel(r'$<\varepsilon>$', fontsize='xx-large')
plt.xlabel(r'$\log_{10} \frac{\sigma}{\sigma^0}$, $\sigma^0 = 1$fb', size='xx-large')
plt.ylim([-0.2,0.2])

xmin = -20
xmax = 7

plt.xlim([xmin, xmax])
plt.xticks(np.arange(xmin, xmax, 2.0))

plt.title(r'Lin+Log $\sigma \geq$ 1e-16')
plt.savefig('/home/ingrid/Documents/Master/ML/Distributed_GP/bananas/plots/1000ppe_1exp_linlog_min16.pdf')

plt.show()
