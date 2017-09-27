"""
For plotting the results of Distributed Gaussian Processes, DGP.py.

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

n_bins = 100


# Infile given from command line
if len(sys.argv) > 1:
    infile = open(sys.argv[1])
else:
    infile = open('results.txt')

numbers = []

# Read infile
for line in infile:
    if not line.startswith("T"):
        line = line.split()
        if line:
            for i in line:
                word = float(i)
                numbers.append(word)


N = int(len(numbers)/4)

errors_GP = np.zeros(N)
errors_DGP = np.zeros(N)
mus_DGP = np.zeros(N)
sigmas_DGP = np.zeros(N)


for i in range(N):
    errors_GP[i] = numbers[4*i]
    errors_DGP[i] = numbers[4*i+1]
    mus_DGP[i] = numbers[4*i+2]
    sigmas_DGP[i] = numbers[4*i+3]
    

# Remove weird values

lims = 0.25
n_bins = int(2*lims*25)
    
#Find the number of points outside desired interval
counter_GP = 0
counter_DGP = 0

index_list_GP = []
index_list_DGP = []

for i in range(len(errors_GP)):
        if abs(errors_GP[i]) >= lims:
            counter_GP += 1
            index_list_GP.append(i)
        if abs(errors_DGP[i]) >= lims:
            counter_DGP +=1
            index_list_DGP.append(i)
            
errors_GP_new = np.delete(errors_GP, index_list_GP)
errors_DGP_new = np.delete(errors_DGP, index_list_DGP)

per_GP = counter_GP/float(N)
per_DGP = counter_DGP/float(N)

#Print to screen the number of points outside desired interval
print "GP: The fraction of points outside (- %.2f, %.2f ) is %.4f" % (lims, lims, per_GP)
print "DGP: The fraction of points outside (- %.2f, %.2f ) is %.4f" % (lims, lims, per_DGP)

print "New number of test points is GP: %.f, DGP: %.f " % (len(errors_GP_new), len(errors_DGP_new))

# Find Gaussian approximations
(m_GP, s_GP) = norm.fit(errors_GP_new)
(m_DGP, s_DGP) = norm.fit(errors_DGP_new)
    
###########################################################
# Plot histograms of GP and DGP                           #
###########################################################


# Change to norm=True to include Gaussian approx
plt.hist(errors_GP_new, bins= n_bins, histtype='step', linewidth=2)
plt.hist(errors_DGP_new, bins= n_bins, histtype='step', linewidth=2)

xt = plt.xticks()[0]
xmin, xmax = min(xt), max(xt)
lnspc = np.linspace(xmin, xmax, len(errors_GP_new))
pdf_gp = stats.norm.pdf(lnspc, m_GP, s_GP)
pdf_dgp = stats.norm.pdf(lnspc, m_DGP, s_DGP)

#plt.plot(lnspc, pdf_gp, label="Norm")
#plt.plot(lnspc, pdf_dgp, label="Norm")
   
#plt.legend(['GP, m: %.3f, s: %.3f' % (m_GP, s_GP), 'DGP, m: %.3f, s: %.3f' % (m_DGP, s_DGP), 'GP', 'DGP'], loc="upper left")

plt.legend(['GP', 'DGP'], loc='best')

plt.xlim(-lims, lims)
plt.title('Relative errors')
plt.xlabel('Relative error')
plt.ylabel('n')
#plt.savefig('DGP_fx.pdf')
plt.show()


######################################################################
# Plot each prediction with error bar                                #
######################################################################

"""
x = np.linspace(0,1,int(N/10))
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


