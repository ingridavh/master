#Takes the file bigapples.txt and divides points according to size
#Plots histograms of relative error

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import seaborn as sns
import matplotlib.axes as mpx

#Read results from file bigapples.txt
myfile = open('bigapples_08.txt')
numbers = []

for line in myfile:
	if not line.startswith("T"):
		line = line.split()
		if line:
			for i in line:
			    word = float(i)
			    numbers.append(word)

N = len(numbers)/5

m2L = np.zeros(N)
M3 = np.zeros(N)
target_predict = np.zeros(N)
target_test = np.zeros(N)
errors = np.zeros(N)

#Pick out input parameters, predicted and test target value, and relative error
for i in range(N):
    m2L[i] = numbers[5*i]
    M3[i] = numbers[5*i+1]
    target_predict[i] = numbers[5*i+2]
    target_test[i] = numbers[5*i+3]
    errors[i] = numbers[5*i+4]

#############################################
# Organize by NLO-size                      #
#############################################

print "Smallest NLO:", min(target_test)
print "Largest NLO: ", max(target_test)


errors_split = []
target_test_split = []


#Rearrange arrays, make a matrix [a][b] where a is exponent, and b are points
for i in range(33):
    temp_list = []
    temp_list_targets = []
    for j in range(N):
        if (target_test[j] >= (i-31)) and (target_test[j] <= (i-30)):
            temp_list.append(errors[j])
            temp_list_targets.append(target_test[j])
    errors_split.append(np.asarray(temp_list))
    target_test_split.append(np.array(temp_list_targets))

#Turn back into arrays
errors_split = np.asarray(errors_split);
target_test_split = np.asarray(target_test_split)


###########################################
# Plot splitted dataset                   #
###########################################

mus = []
sigmas = []

plt.figure(1)
for i in range(33):
    if len(errors_split[i]) >= 1:
        (mu, sigma) = norm.fit(errors_split[i])
        mus.append(mu); sigmas.append(sigma)
    else:
        mu = 0
        sigma = 0
        mus.append(mu); sigmas.append(sigma)

mus = np.asarray(mus)
sigmas = np.asarray(sigmas)

#Print table of values
#print "%-24s %-10s %-10s" % ("Power", "Mu", "Sigma")
#for i in range(33):
        #print " %-4i to %-10i  %-10.4f %10.4f" % ((i-31), (i-30), mus[i], sigmas[i])


#Plot histograms. Remember indices [1,2,3] gave no result and are set to zero.
#plt.subplot(3,1,1)

####Testing values----------------------------------------
#0 is a LOT off, has values at -60000

#plt.subplot(2,1,1)
#plt.title('Relative difference between predicted and test target values', size='xx-large')
#plt.hist(errors_split[0], normed=True, bins=10, histtype='step', linewidth=1)
#plt.legend(['0'])
#plt.ylabel('n', size='x-large')

#4 is a little off, has values at -140
#plt.subplot(2,1,2)
#plt.hist(errors_split[4], normed=True, bins=10, histtype='step', color='salmon', linewidth=1)
#plt.legend(['4'])
#5 is quite close, has values at -12
#plt.hist(errors_split[5], normed=True, bins=10, histtype='step', linewidth=1)
####------------------------------------------------------



########################################################################
# Plot histograms of all values                                        #
########################################################################

#legend_list = []
#for i in range(5,33):
#        plt.hist(errors_split[i], normed=True, bins=10, histtype='step', linewidth=1)
#        legend_list.append("%i" % (i-30))

#plt.legend(legend_list)
#plt.title('Relative difference between predicted and test target values', size='xx-large')

#plt.xlabel('rel diff', size='x-large')
#plt.ylabel('n', size='x-large')


########################################################################
# Plot histograms of interesting values                                #
########################################################################

"""
plt.subplot(2,2,1)
legend_list = []
for i in range(10,15):
        plt.hist(errors_split[i], normed=True, bins=10, histtype='step', linewidth=1.5)
        legend_list.append("%i" % (i-30))

plt.legend(legend_list, loc="North West")
plt.title('Relative difference', size='xx-large')
plt.ylabel('n', size='x-large')

plt.subplot(2,2,2)
legend_list = []
for i in range(16,20):
        plt.hist(errors_split[i], normed=True, bins=10, histtype='step', linewidth=1.5)
        legend_list.append("%i" % (i-30))

plt.legend(legend_list, loc="North West")

plt.subplot(2,2,3)
legend_list = []
for i in range(20,25):
        plt.hist(errors_split[i], normed=True, bins=10, histtype='step', linewidth=1.5)
        legend_list.append("%i" % (i-30))

plt.legend(legend_list, loc="North West")
plt.xlabel('rel diff', size='x-large')
plt.ylabel('n', size='x-large')

plt.subplot(2,2,4)
legend_list = []
for i in range(25,33):
        plt.hist(errors_split[i], normed=True, bins=10, histtype='step', linewidth=1.5)
        legend_list.append("%i" % (i-30))

plt.legend(legend_list, loc="upper left")
plt.xlabel('rel diff', size='x-large')
#plt.savefig('Plots/test_split_error_hist_Abel_8000p_comparted.pdf')
"""

########################################################################
# Plot sigma and mu                                                    #
########################################################################

x = np.linspace(-31,1,33)
"""
plt.subplot(2,1,1)
plt.plot(x[5:], sigmas[5:], color='blue')
#plt.plot(x, sigmas, color='blue')
plt.title('Gaussian fitted parameters of histograms', size='x-large')
plt.legend(['Sigma'])
plt.ylabel('Sigma', size='large')

plt.subplot(2,1,2)
plt.plot(x[5:], mus[5:], color='salmon')
#plt.plot(x, mus, color='salmon')
plt.legend(['Mu'])
plt.xlabel('Log(NLO)', size='large')
plt.ylabel('Mu', size='large')

plt.savefig('test_split_gaussfit_Abel_8000p_interest.pdf')
"""

#######################################################################
# Plot with errorbars                                                 #
#######################################################################
cut_int = 24
yerr = sigmas[cut_int:]
yerr_ = np.zeros(len(x[cut_int:]))+1
plt.errorbar(x[cut_int:],mus[cut_int:], yerr=yerr)
plt.plot(x[cut_int:], mus[cut_int:], 'o', color='salmon')
plt.title('Mean error with std', size='xx-large')
plt.ylabel('Mean error', size='large')
plt.xlabel('log(NLO)', size='large')
#plt.savefig('Plots/Abel_8000p_meanerror.pdf')
plt.show()
