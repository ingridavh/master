#Takes the file bigapples.txt and divides points according to size
#Plots histograms of relative error

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import seaborn as sns
import matplotlib.axes as mpx
import matplotlib as mpl
import sys

#sns.palplot(sns.color_palette("Paired"))
mpl.style.use('ingrid_style')


# Take command line arguments
if len(sys.argv) >= 2:
        myfile = open(sys.argv[1])
else:
        print "Error! No input file was provided."
        sys.exit(1)

if len(sys.argv) >= 3:
        plot_save = sys.argv[2]
else:
        plot_save = False

#Read results from infile
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

print int(min(target_test))
print int(max(target_test))

min_pow = int(min(target_test))
max_pow = int(max(target_test))

num_splits = max_pow - min_pow
print "There are ", num_splits, " powers" 

errors_split = []
target_test_split = []

#Rearrange arrays, make a matrix [a][b] where a is exponent, and b are points
for i in range(num_splits+2):
    temp_list = []
    temp_list_targets = []
    for j in range(N):
        if (target_test[j] >= (i-(num_splits+1)+max_pow)) and (target_test[j] <= (i-(num_splits)+max_pow)):
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

for i in range(num_splits+2):
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
#for i in range(num_splits+2):
#        print " %-4i to %-10i  %-10.4f %10.4f" % ((i-num_splits+max_pow-1), (i-num_splits+max_pow), mus[i], sigmas[i])


#######################################################################
# Plot histogram of errors                                            #
#######################################################################

"""
lims = 1.5
n_bins = 2*lims*25
    
#Find the number of points outside desired interval
counter = 0
index_list = []
for i in range(len(errors)):
        if abs(errors[i]) >= lims:
                counter += 1
                index_list.append(i)
errors_new = np.delete(errors, index_list)

per = counter/float(N)

#Print to screen the number of points outside desired interval
print "The fraction of points outside (- %.2f, %.2f ) is %.4f" % (lims, lims, per)
"""

"""
# Gaussian approximation
(m, s) = norm.fit(errors_new)

plt.hist(errors_new, bins=int(n_bins), normed=True, color='salmon')
plt.title("Relative difference between predicted and test target", size='xx-large')
plt.xlabel('n', size='x-large')
plt.ylabel('rel diff', size='x-large')

xt = plt.xticks()[0]
xmin, xmax = min(xt), max(xt)
lnspc = np.linspace(xmin, xmax, len(errors_new))
pdf_g = norm.pdf(lnspc, m, s)
plt.plot(lnspc, pdf_g, label="Norm")
plt.legend(["mu = %.3f, sigma = %.3f" % (m,s)], fontsize= 'large', loc='best')
#plt.savefig('Plots/meanerror_gaussian_fit_cut.pdf')
plt.show()
"""

########################################################################
# Plot histograms of all values                                        #
########################################################################

"""
legend_list = []
for i in range(5,33):
        plt.hist(errors_split[i], normed=True, bins=10, histtype='step', linewidth=1)
        legend_list.append("%i" % (i-30))

plt.legend(legend_list)
plt.title('Relative difference between predicted and test target values', size='xx-large')

plt.xlabel('rel diff', size='x-large')
plt.ylabel('n', size='x-large')
plt.show()
"""

########################################################################
# Plot histograms of interesting values                                #
########################################################################
"""
plt.subplot(2,2,1)
legend_list = []
for i in range(10,15):
        plt.hist(errors_split[i], normed=True, bins=10, histtype='step', linewidth=1.5)
        legend_list.append("%i" % (i-30))

plt.legend(legend_list, loc="best")
plt.title('Relative difference', size='xx-large')
plt.ylabel('n', size='x-large')
plt.xlim(-5,5)

plt.subplot(2,2,2)
legend_list = []
for i in range(16,20):
        plt.hist(errors_split[i], normed=True, bins=10, histtype='step', linewidth=1.5)
        legend_list.append("%i" % (i-30))

plt.legend(legend_list, loc="best")
plt.xlim(-5,5)

plt.subplot(2,2,3)
legend_list = []
for i in range(20,25):
        plt.hist(errors_split[i], normed=True, bins=10, histtype='step', linewidth=1.5)
        legend_list.append("%i" % (i-30))

plt.legend(legend_list, loc="best")
plt.xlabel('rel diff', size='x-large')
plt.ylabel('n', size='x-large')

plt.subplot(2,2,4)
legend_list = []
for i in range(25,33):
        plt.hist(errors_split[i], normed=True, bins=10, histtype='step', linewidth=1.5)
        legend_list.append("%i" % (i-30))

plt.legend(legend_list, loc="upper left")
plt.xlabel('rel diff', size='x-large')
plt.xlim(-1.5,1.5)


#plt.savefig('Plots/errors_split_data.pdf')
plt.show()
"""

#######################################################################
# Plot mus with errorbars                                             #
#######################################################################

x = np.linspace(min_pow,max_pow+1,num_splits+2)
print x

cut_int = 24
yerr = sigmas[cut_int:]
yerr_ = np.zeros(len(x[cut_int:]))+1
plt.plot(x[cut_int:], mus[cut_int:], 'o', linewidth=2)
plt.errorbar(x[cut_int:], mus[cut_int:], yerr=yerr, linewidth=1.0)
plt.title('Mean error with std', size='xx-large')
plt.ylabel('Mean error', size='large')
plt.xlabel('log(NLO)', size='large')
plt.ylim([-0.8,0.6])
if not (plot_save == False):
        plt.savefig(plot_save)
plt.show()

#######################################################################
# Plot histograms of interesting values                               #
#######################################################################
"""
legend_list = []
for i in range(16,num_splits+2):
        if i != 19 and i != 16:
                plt.hist(errors_split[i], normed=True, bins=10, histtype='step', linewidth=1.5)
                legend_list.append("%i" % (i-num_splits+max_pow))

plt.legend(legend_list, loc="upper left")
plt.xlim([-1,1])
plt.title('Relative difference', size='xx-large')
plt.ylabel('n', size='x-large')
plt.show()
"""
