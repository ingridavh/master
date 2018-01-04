from dgp import dgp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy import stats
from scipy.stats import norm
import matplotlib as mpl

mpl.style.use('ingrid_style')

x = abs(np.random.randn(10000,2))*10
def f(x):
    return 4*x[:,0]*x[:,1]

y = f(x)

outfile_name = 'ut.txt'

kernel1 = RBF(length_scale=1, length_scale_bounds=(1e-02, 1e05)) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-03, 1000.0))

my_dgp = dgp(4, outfile_name, kernel=kernel1)
my_dgp.fit(x, y, trainsize=0.5)

# Read outfile and plot results

myfile = open(outfile_name)

numbers = []

# Read infile
for line in myfile:
    if not line.startswith("T"):
        line = line.split()
        if line:
            for i in line:
                word = float(i)
                numbers.append(word)


N = int(len(numbers)/4)

errors = np.zeros(N)
mus = np.zeros(N)
sigmas = np.zeros(N)


for i in range(N):
    errors[i] = numbers[3*i]
    mus[i] = numbers[3*i+1]
    sigmas[i] = numbers[3*i+2]

    
# Plot histogram of errors

lims = 0.25
n_bins = int(2*lims*25)
    
#Find the number of points outside desired interval
counter = 0
index_list = []
for i in range(len(errors)):
        if abs(errors[i]) >= lims:
                counter += 1
                index_list.append(i)
errors_new = np.delete(errors, index_list)

#Print to screen the number of points outside desired interval
print "The fraction of points outside (-2,2) is %.4f" % float(counter/float(N))


################################################################
# Plot results with Gaussian fit                               #
################################################################

(mu, sigma) = norm.fit(errors_new)

#Plot with Gaussian fit
plt.hist(errors_new, normed=True, bins=int(n_bins), color='salmon')
xt = plt.xticks()[0]
xmin, xmax = min(xt), max(xt)
lnspc = np.linspace(xmin, xmax, len(errors_new))
pdf_g = stats.norm.pdf(lnspc, mu, sigma)
plt.plot(lnspc, pdf_g, label="Norm")


#Make plot pretty
plt.title('Relative difference between predicted and test target values', size='xx-large')
plt.xlabel('rel diff', size='x-large')
plt.ylabel('n', size='x-large')
plt.legend(["mu = %.3f, sigma = %.3f" % (mu,sigma)], fontsize= 'large', loc='best')
#plt.savefig('Plots/meanerror_gaussian_fit.pdf')
plt.show()

