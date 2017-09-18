import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.pylab as ply

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



#Find non-relative, and non-log error
errors_nonrel = errors*np.power(10,target_test)
errors_nonlog = (target_test-target_predict)/target_test

#Find the number of points outside desired interval
counter = 0
for err in errors:
        if abs(err) >= 2:
                counter += 1
per = counter/float(N)
print "The fraction of points outside (-2,2) is %.4f" % per

"""
#Find mean (mu) and standard deviation (sigma)
(mu, sigma) = norm.fit(errors)

#Plot results

#Plot histogram

plt.hist(errors, bins = np.linspace(-2,2, 100), norm=True)

#Plot with Gaussian
#Find mean (mu) and standard deviation (sigma)
(mu, sigma) = norm.fit(errors)

plt.figure(1)
n, bins, patches = plt.hist(errors, 100, normed=1, facecolor='blue', align='mid')
y = mlab.normpdf(bins,mu,sigma)
plt.plot(bins, y, 'r--', linewidth=2)


plt.title('Relative difference between predicted and test target values', size='xx-large')
plt.xlabel('rel diff', size='x-large')
plt.ylabel('n', size='x-large')
plt.savefig('error_True_8000p_fittedparams_Abel_gf.pdf')
plt.show()
"""

#Method 2
"""
ply.hist(errors, normed=True)
xt = ply.xticks()[0]
xmin, xmax = min(xt), max(xt)
lnspc = np.linspace(xmin, xmax, len(errors))
#lnspc = np.linspace(-2,2,100)
m, s = stats.norm.fit(errors)
pdf_g = stats.norm.pdf(lnspc, m, s)
ply.plot(lnspc, pdf_g, label="Norm")

ply.show()
"""
