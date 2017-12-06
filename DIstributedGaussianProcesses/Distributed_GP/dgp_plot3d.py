"""
For plotting the results of Distributed Gaussian Processes, 
calulated using the class dgp.py.

@author: Ingrid A V Holm
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    my_kernel = str(sys.argv[2])
    
if len(sys.argv) >= 4:
    savefile = sys.argv[3]
else:
    savefile = False


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
msq = np.zeros(N)
mg = np.zeros(N)

for i in range(N):
    errors[i] = numbers[6*i]
    mus[i] = numbers[6*i+1]
    sigmas[i] = numbers[6*i+2]
    y_test[i] = numbers[6*i+3]
    msq[i] = numbers[6*i+4]
    mg[i] = numbers[6*i+5]

# Define mass parameter from calc
m_sq = mg**2-msq**2

# Choose limits for Gaussian fits
plotlims = 1.0
lims1 = 0.5
lims2 = 0.25
n_bins_better = int(2*plotlims*25)
    

###########################################################
# Set kernel                                              #
###########################################################

if my_kernel == 'kernel6':
    plt.title('Relative error, $k=C_1 \exp(-x^2/l^2)+ C_2$')
elif my_kernel == 'kernel7':
    plt.title('Relative error, $k=C_1 \exp(-x^2/l^2)+ C_2 \epsilon_{WK}$')
elif my_kernel == 'kernel_ell':
    plt.title(r'Relative error, $k=C_1 \exp(-(x^T M x))+ C_2$')


plt.xlabel('Relative error')
plt.ylabel('n')
if not savefile == False:
    plt.savefig(savefile)



######################################################################
# Plot 3D scatter plot                                               #
######################################################################

f1 = plt.figure(1)
ax1 = f1.add_subplot(111, projection='3d')
ax1.scatter(msq, mg, y_test)


plt.show()

