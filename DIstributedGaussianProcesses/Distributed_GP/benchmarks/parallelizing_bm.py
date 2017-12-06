"""
 Benchmark test script to test the parallelizing algorithm to be used 
 on the module 'dgp_parallel'. MPI for Python is used for parallelizing, 
 and number of experts and number of nodes are given as command line 
 arguments. The program takes the following arguments

 mpiexec -n 'number of nodes' python thisprogram.py 'number of experts'
"""
import numpy as np
import sys
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if len(sys.argv) >= 2:
    n_exp = int(sys.argv[1])
else:
    print 'Error! No number of experts was provided.'
    sys.exit(0)

if rank == 0:
    print "Working with %.f experts, and %.f nodes" % (n_exp, size)

# Divide experts between nodes
if n_exp%size == 0:
    local_n = n_exp/size
else:
    # FIX THIS! Not yet figured out! 
    print 'Error! Number of experts must be a multiple of n'
    sys.exit(1)

###########################################################
# Begin routine                                           #
###########################################################
N = 20
M = 40

#x_train = np.random.randn(M)
x_train = np.linspace(1, M, M)
x_test = np.linspace(1, N, N)

# Divide x's into subsets
x_train_subsets = np.split(x_train, n_exp)

h = 1 # Step size
local_min = rank*local_n*h
local_max = local_min + local_n*h
    
def f(x):
    return np.sum(x)

def g(x, y):
    return y*f(x)

print local_min, local_max, local_n
def kjor(x_subsets, y, local_min, local_max):
    kjor_ut = np.zeros((n_exp, N))
    for i in range(local_min, local_max, h):
        x = x_subsets[i]
        kjor_ut[i] += g(x, y)
    return kjor_ut
        
experts_fit = np.zeros((n_exp, N))
total = np.zeros((n_exp, N))

experts_fit = kjor(x_train_subsets, x_test, local_min, local_max)

# Send all results to rank 0
comm.Reduce(experts_fit, total, op=MPI.SUM, root=0)

if comm.rank == 0:
    print x_train_subsets
    print total





