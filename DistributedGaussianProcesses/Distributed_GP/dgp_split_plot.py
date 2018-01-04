#Takes the file bigapples.txt and divides points according to size
#Plots histograms of relative error

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.axes as mpx
import numpy as np
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
        plot_save = sys.argv[3]
else:
        plot_save = False


# To modify histograms to stay inside limits [-1,1]
mod = True
lims = 1

#Read results from infile
numbers = []

for line in myfile:
	if not line.startswith("T"):
		line = line.split()
		if line:
			for i in line:
			    word = float(i)
			    numbers.append(word)

N = int(len(numbers)/6)

errors = np.zeros(N)
mus = np.zeros(N)
sigmas = np.zeros(N)
y_test = np.zeros(N)
m1 = np.zeros(N)
m2 = np.zeros(N)

for i in range(N):
    errors[i] = numbers[6*i]
    mus[i] = numbers[6*i+1]
    sigmas[i] = numbers[6*i+2]
    y_test[i] = numbers[6*i+3]
    m1[i] = numbers[6*i+4]
    m2[i] = numbers[6*i+5]

# Choose limits for Gaussian fits
plotlims = 1.0
lims1 = 0.5
lims2 = 0.25
n_bins_better = int(2*plotlims*25)

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
print "%-24s %-10s %-10s %-10s" % ("Power", "Mu", "Sigma", "Number of points")
for i in range(num_splits+2):
        print " %-4i to %-10i  %-10.4f %10.4f %10.4f" % ((i-num_splits+max_pow-1), (i-num_splits+max_pow), mu_array[i], sigma_array[i], len(errors_split[i]))


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
for i in range(25,num_splits):
        print i-33
        plt.hist(errors_split[i], normed=True, bins=10, histtype='step', linewidth=1)
        legend_list.append("%i" % (i+min_pow))

plt.legend(legend_list)
plt.title('Relative difference between predicted and test target values', size='xx-large')

plt.xlim([-5,5])
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
for i in range(1,15):
        plt.hist(errors_split[i], normed=True, bins=10, histtype='step', linewidth=1.5)
        legend_list.append("%i" % (i+min_pow))

plt.legend(legend_list, loc="best")
plt.title('Relative difference', size='xx-large')
plt.ylabel('n', size='x-large')
plt.xlim(-5,5)

plt.subplot(2,2,2)
legend_list = []
for i in range(16,20):
        plt.hist(errors_split[i], normed=True, bins=10, histtype='step', linewidth=1.5)
        legend_list.append("%i" % (i+min_pow))

plt.legend(legend_list, loc="best")
plt.xlim(-5,5)

plt.subplot(2,2,3)
legend_list = []
for i in range(20,25):
        plt.hist(errors_split[i], normed=True, bins=10, histtype='step', linewidth=1.5)
        legend_list.append("%i" % (i+min_pow))

plt.legend(legend_list, loc="best")
plt.xlabel('rel diff', size='x-large')
plt.ylabel('n', size='x-large')

plt.subplot(2,2,4)
legend_list = []
for i in range(25,33):
        plt.hist(errors_split[i], normed=True, bins=10, histtype='step', linewidth=1.5)
        legend_list.append("%i" % (i+min_pow))

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

cut_int = 10
yerr = sigma_array[cut_int:]
yerr_ = np.zeros(len(x[cut_int:]))+0.1
plt.plot(x[cut_int:], mu_array[cut_int:], 'o', linewidth=2)
plt.errorbar(x[cut_int:], mu_array[cut_int:], yerr=yerr, linewidth=2.0)
plt.errorbar(x[cut_int:], mu_array[cut_int:], yerr=yerr_, linewidth=2.0, alpha=0.5)
plt.title('Mean error with std', size='xx-large')
plt.ylabel('Mean error', size='large')
plt.xlabel('log(NLO)', size='large')
plt.ylim([-0.8,0.6])
if not (plot_save == False):
        plt.savefig(plot_save)
#plt.savefig('/home/ingrid/Documents/Master/ML/Distributed_GP/tester_log_lin/plots/lin_kell3_05.pdf')
plt.show()

#######################################################################
# Plot histograms of interesting values with double histograms        #
#######################################################################


esn = [] # Errors_Split_New

for i in range((num_splits+2)/2):
        temp = np.append(errors_split[2*i],errors_split[2*i+1])
        esn.append(temp)

esn = errors_split
esn = np.array(esn)
esn_new = []

if mod == True:
        index_list = []
        for i in range(len(esn)):
                eps = esn[i]
                for j in range(len(eps)):
                        if abs(eps[j]) > 1:
                                index_list.append(j)
                esn_ = np.delete(esn[i], index_list)
                esn_new.append(esn_)

else:
        esn_new = esn
                
N = len(esn_new)
mus_esn = []
sigmas_esn = []

# Plot histograms

"""
for i in range(N-1):
        legend_list = []
        plotmin = i
        plotmax = i+1
        # Find Gaussian approximation and plot it

        if len(esn_new[i]) > 1:
                for j in range(plotmin, plotmax):
                        # Find Gaussian fit for this interval

                        xmin = np.min(esn_new[j])
                        xmax = np.max(esn_new[j])
                        
                        (mu, sigma) = norm.fit(esn_new[j])
                        mus_esn.append(mu)
                        sigmas_esn.append(sigma)

                        lnspc = np.linspace(xmin, xmax, 100)
                        pdf_g = norm.pdf(lnspc, mu, sigma)
                        plt.plot(lnspc, pdf_g, linewidth= 2.0)

                        legend_list.append("[%i, %i]: $\mu$= %.3f, $\sigma$= %.3f" % ((j-num_splits+max_pow-1), (j-num_splits+max_pow), mu, sigma))




                # Plot histogram
                for j in range(plotmin, plotmax):
                        plt.hist(esn_new[j], normed=True, bins=40, linewidth=0.8)
                        legend_list.append("[%i, %i]" % (2*j-num_splits+max_pow, 2*j+1-num_splits+max_pow))


                plt.legend(legend_list, loc="upper left")
                plt.xlim([-1,1])
                plt.title('Relative difference (%i points)' % len(esn_new[j]), size='xx-large')
                plt.ylabel('n', size='x-large')
                plt.show()
"""


