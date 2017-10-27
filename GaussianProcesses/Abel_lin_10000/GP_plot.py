import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.pylab as ply
from scipy.stats import norm

import seaborn as sns

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
print "The fraction of points outside (-2,2) is %.4f" % per
print "The indices of these objects are", index_list


################################################################
# Plot results with Gaussian fit                               #
################################################################

#Plot histogram
#plt.hist(errors, bins = np.linspace(-2,2, 100), norm=True)

#Plot with Gaussian
#Find mean (mu) and standard deviation (sigma)
(mu, sigma) = norm.fit(errors_new)

print "Errors follow a Gaussian distribution with mean mu=%.5f, and std sigma=%.5f" % (mu, sigma)

plt.figure(1)


#Plot with Gaussian fit
ply.hist(errors_new, normed=True, bins=int(n_bins), color='salmon')
xt = ply.xticks()[0]
xmin, xmax = min(xt), max(xt)
lnspc = np.linspace(xmin, xmax, len(errors_new))
pdf_g = stats.norm.pdf(lnspc, mu, sigma)
ply.plot(lnspc, pdf_g, label="Norm")


#Make plot pretty
plt.title('Relative difference between predicted and test target values', size='xx-large')
plt.xlabel('rel diff', size='x-large')
plt.ylabel('n', size='x-large')
plt.legend(["mu = %.3f, sigma = %.3f" % (mu,sigma)], fontsize= 'large', loc='best')
#plt.savefig('Plots/meanerror_gaussian_fit.pdf')
ply.show()

