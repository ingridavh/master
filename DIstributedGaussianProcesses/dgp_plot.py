"""
For plotting the results of Distributed Gaussian Processes, 
calulated using the class dgp.py.

@author: Ingrid A V Holm
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys

# Import stats
from scipy import stats
from scipy.stats import norm

mpl.style.use('ingrid_style')

# Infile given from command line
if len(sys.argv) >= 2:
    infile = open(sys.argv[1])
else:
    infile = open('results.txt')

if len(sys.argv) >= 3:
    savefile = sys.argv[2]
else:
    savefile = False

if len(sys.argv) >= 4:
    my_kernel = str(sys.argv[3])

numbers = []

# Read infile
for line in infile:
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
    
#Find the number of points outside desired interval
counter_1 = 0
counter_2 = 0
counter_plot = 0
index_list_1 = []
index_list_2 = []
index_list_plot = []

for i in range(N):
        if abs(errors[i]) >= lims1:
            counter_1 +=1
            index_list_1.append(i)
        if abs(errors[i]) >= lims2:
            counter_2 += 1
            index_list_2.append(i)
        # Add one more so plotting gets more normal
        if abs(errors[i]) >= plotlims:
            counter_plot += 1
            index_list_plot.append(i)
            
errors_new_1 = np.delete(errors, index_list_1)
errors_new_2 = np.delete(errors, index_list_2)
errors_plot = np.delete(errors, index_list_plot)


per_1 = counter_1/float(N)
per_2 = counter_2/float(N)
per_plot = counter_plot/float(N)

#Print to screen the number of points outside desired interval

print "DGP: The fraction of points outside (- %.2f, %.2f ) is %.4f" % (lims1, lims1, per_1)
print "DGP: The fraction of points outside (- %.2f, %.2f ) is %.4f" % (lims2, lims2, per_2)
print "DGP: The fraction of points outside (- %.2f, %.2f ) is %.4f" % (plotlims, plotlims, per_plot)
print "New number of test points is  %.f " % len(errors_new_1)

# Find Gaussian approximations

(m_1, s_1) = norm.fit(errors_new_1)
(m_2, s_2) = norm.fit(errors_new_2)

###########################################################
# Plot histograms of GP and DGP                           #
###########################################################
"""
# Change to norm=True to include Gaussian approx
plt.hist(errors_plot, bins = n_bins_better, normed=True)

plt.legend(['DGP'], loc='best')
#plt.xlim(-plotlims, plotlims)
plt.title('Relative errors')
plt.xlabel('Relative error')
plt.ylabel('n')
#plt.savefig('Abel_20k_4experts/DGP_pears_05.pdf')
plt.show()
"""

# Plot histograms with Gaussian approximations


plt.hist(errors_plot, normed = True, bins= n_bins_better, linewidth=2)

xt = plt.xticks()[0]
xmin, xmax = min(xt), max(xt)
lnspc = np.linspace(xmin, xmax, len(errors_plot))
pdf_1 = stats.norm.pdf(lnspc, m_1, s_1)
pdf_2 = stats.norm.pdf(lnspc, m_2, s_2)

plt.plot(lnspc, pdf_1, lnspc, pdf_2)   
plt.legend(['$\mu$: %.3f, $\sigma^2$: %.3f (0.50)' % (m_1, s_1),'$\mu$: %.3f, $\sigma^2$: %.3f (0.25)' % (m_2, s_2)], loc='best')

plt.xlim(-plotlims, plotlims)
if my_kernel == 'kernel1':
    plt.title('Relative error, $k=C_1 \exp(-x^2/l^2)$')
elif my_kernel == 'kernel4':
    plt.title('Relative error, $k=\exp(-x^2/l^2)+ \epsilon_{WK}$')
elif my_kernel == 'kernel5':
    plt.title('Relative error, $k=C_1 \exp(-x^2/l^2)+ \epsilon_{WK}$')
elif my_kernel == 'kernel6':
    plt.title('Relative error, $k=C_1 \exp(-x^2/l^2)+ C_2$')
elif my_kernel == 'kernel7':
    plt.title('Relative error, $k=C_1 \exp(-x^2/l^2)+ C_2 \epsilon_{WK}$')



plt.xlabel('Relative error')
plt.ylabel('n')
if not savefile == False:
    plt.savefig(savefile)
plt.show()


######################################################################
# Plot each prediction with error bar                                #
######################################################################

m1_index_sort = np.argsort(m1)
m2_index_sort = np.argsort(m2)

m1_sort = np.zeros(N)
m2_sort = np.zeros(N)
y_test_sort_m1 = np.zeros(N)
y_test_sort_m2 = np.zeros(N)
y_pred_sort_m1 = np.zeros(N)
y_pred_sort_m2 = np.zeros(N)
sigma_sort_m1 = np.zeros(N)
sigma_sort_m2 = np.zeros(N)

for i in range(N):
    m1_sort[i] = m1[m1_index_sort[i]]
    m2_sort[i] = m2[m2_index_sort[i]]

    y_test_sort_m1[i] = y_test[m1_index_sort[i]]
    y_test_sort_m2[i] = y_test[m2_index_sort[i]]

    y_pred_sort_m1[i] = mus[m1_index_sort[i]]
    y_pred_sort_m2[i] = mus[m2_index_sort[i]]

    sigma_sort_m1[i] = sigmas[m1_index_sort[i]]
    sigma_sort_m2[i] = sigmas[m2_index_sort[i]]
                        
yerr_m1 = np.log10(sigma_sort_m1)
yerr_m2 = np.log10(sigma_sort_m1)

"""
plt.plot(m1_sort[0::100], y_pred_sort_m1[0::100])
plt.plot(m1_sort[0::100], y_test_sort_m1[0::100], 'o')
#plt.errorbar(m1_sort[0::200], y_pred_sort_m1[0::200], yerr=np.log10(sigma_sort_m1[0::200]))
#plt.fill_between(m1_sort[0::200],  y_pred_sort_m1[0::200] - yerr_m1[0::200], y_pred_sort_m1[0::200] + yerr_m1[0::200], alpha=0.5)
plt.show()

plt.plot(m2_sort[0::100], y_pred_sort_m2[0::100])
plt.plot(m2_sort[0::100], y_test_sort_m2[0::100], 'o')
plt.show()
"""

"""
x = np.linspace(0, 1, int(N/10))
y = np.zeros(N/10)
yerr = np.zeros(N/10)

for i in range(N/10):
    y[i] = mus_DGP[i*10]
    yerr[i] = sigmas_DGP[i*10]
#y = mus_DGP
#yerr = sigmas_DGP

print N

#plt.errorbar(x, y, yerr=yerr)
#plt.fill_between(x, y-yerr, y+yerr)
plt.plot(x, y, '-', color='salmon')
#plt.plot(x, variances[0], x, variances[1], x, variances[2])
plt.title('Mean error with std')
plt.ylabel('Mean error')
plt.xlabel('log(NLO)')
plt.show()
"""


