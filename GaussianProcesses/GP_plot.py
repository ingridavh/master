import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.pylab as ply
from scipy.stats import norm
import sys

import matplotlib as mpl

mpl.style.use('ingrid_style')
#import seaborn as sns

#Read results from infile

if len(sys.argv) >= 2:
        myfile = open(sys.argv[1])
else:
        print "Error! You must provide a file to read."
        sys.exit(0)

if len(sys.argv) >= 3:
        plot_save = sys.argv[2]
else:
        plot_save = False

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


lims = 0.5
lims2 = 0.25
n_bins = 4*lims*25
    
#Find the number of points outside desired interval
counter = 0
counter2 = 0
index_list = []
index_list_2 = []

for i in range(len(errors)):
        if abs(errors[i]) >= lims:
                counter += 1
                index_list.append(i)
        if abs(errors[i]) >= lims2:
                counter2 += 1
                index_list_2.append(i)
errors_new = np.delete(errors, index_list)
errors_new_2 = np.delete(errors, index_list_2)

per = counter/float(N)
per2 = counter2/float(N)

#Print to screen the number of points outside desired interval
print "The fraction of points outside (- %.2f, %.2f ) is %.4f" % (lims, lims, per)
print "The fraction of points outside (- %.2f, %.2f ) is %.4f" % (lims2, lims2, per2)


################################################################
# Plot results with Gaussian fit                               #
################################################################

#Plot histogram
#plt.hist(errors, bins = np.linspace(-2,2, 100), norm=True)

#Plot with Gaussian
#Find mean (mu) and standard deviation (sigma)
(mu, sigma) = norm.fit(errors_new)
(mu2, sigma2) = norm.fit(errors_new_2)

print "Lim=%.2f: Errors follow a Gaussian distribution with mean mu=%.5f, and std sigma=%.5f" % (lims, mu, sigma)
print "Lim=%.2f: Errors follow a Gaussian distribution with mean mu=%.5f, and std sigma=%.5f" % (lims2, mu2, sigma2)

"""
#Plot with Gaussian fit
plt.hist(errors_new, normed=True, bins=int(n_bins), linewidth=0.5)
xt = plt.xticks()[0]
xmin, xmax = min(xt), max(xt)
lnspc = np.linspace(xmin, xmax, len(errors_new))
pdf_g = stats.norm.pdf(lnspc, mu, sigma)
pdf_g_2 = stats.norm.pdf(lnspc, mu2, sigma2)
plt.plot(lnspc, pdf_g, label="Norm", linewidth=2)
plt.plot(lnspc, pdf_g_2, linewidth=2)


#Make plot pretty
plt.title('Relative error, $k= \exp(-x^2/l^2) + \epsilon_{KW}$', size='xx-large')
plt.xlabel('rel diff', size='x-large')
plt.ylabel('n', size='x-large')
plt.legend(['$\mu$: %.3f, $\sigma^2$: %.3f (%.2f)' % (mu, sigma, lims), '$\mu$: %.3f, $\sigma^2$: %.3f (%.2f )' % (mu2, sigma2, lims2)])
if not (plot_save == False):
        plt.savefig(plot_save)
ply.show()
"""


# Visualize results

m2L_sort_index = np.argsort(m2L)
M3_sort_index = np.argsort(M3) 

m2L_sort = np.zeros(N)
M3_sort = np.zeros(N)
target_predict_m2L = np.zeros(N)
target_test_m2L = np.zeros(N)
target_predict_M3 = np.zeros(N)
target_test_M3 = np.zeros(N)


for i in range(N):
        m2L_sort[i] = m2L[m2L_sort_index[i]]
        target_predict_m2L[i] = target_predict[m2L_sort_index[i]]
        target_test_m2L[i] = target_test[m2L_sort_index[i]]
        M3_sort[i] = M3[M3_sort_index[i]]
        target_predict_M3 =  target_predict[M3_sort_index[i]]
        target_test_M3[i] = target_test[M3_sort_index[i]]

plt.plot(m2L_sort[0::200], target_test_m2L[0::200], 'o', linewidth=4)
plt.plot(m2L_sort[0::200], target_predict_m2L[0::200], 'o')
plt.show()
