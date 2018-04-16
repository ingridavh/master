"""
 Created on 19.10.2017

 Reads data files output from harvest_slha.py using pandas and performs ML
 regression with sklearn. Currently uses parallelized Gaussian Processes,
 implemented in the module pdg_parallel.

 To run the script mpi must be used: 

python DGP_phys.py trainsize outfile experts nodes feature_list target_list

 @author Ingrid Angelica Vazquez Holm
"""

from dgp_parallel import dgp_parallel
import sys
import numpy as np
import sklearn as sk
from sklearn.gaussian_process.kernels import RationalQuadratic, RBF, WhiteKernel, ExpSineSquared, DotProduct, Matern, ConstantKernel as C

import matplotlib.pyplot as plt
import pandas as pd

# Set training size
if len(sys.argv) >= 2:
    trainsize = float(sys.argv[1])
    if trainsize >= 1:
        trainsize = int(trainsize)
else:
    trainsize = 0.001

# Set name of outfile
if len(sys.argv) >= 3:
    outfile = sys.argv[2]
else:
    print "Error! No outfile was provided."
    sys.exit(0)

# Choose number of experts
if len(sys.argv) >= 4:
    n_experts = int(sys.argv[4])
else:
    n_experts = 3

if len(sys.argv) >= 5:
    my_njobs = int(sys.argv[4])
else:
    my_njobs = 1

if len(sys.argv) >= 6:
    if int(sys.argv[5]) == 2:
        feature_list = [sys.argv[6], sys.argv[7]]
        target_list = sys.argv[8]
    elif int(sys.argv[5]) == 3:
        feature_list = [sys.argv[6], sys.argv[7], sys.argv[8]]
        target_list = sys.argv[9]
else:
    print "Error! No feature list was provided!"
    sys.exit(1)


print "Size of the training set is %.3f of the total dataset." % trainsize
print "Feature list: ", feature_list
print "Target list: ", target_list

#################################################################
# Read in and treat data (mostly) using pandas.                 #
#################################################################

# Read in data (last column is BS)
data_lin = pd.read_csv('/home/ingrid/Documents/Master/ML/Abel_all_crossections/data_40k.dat', sep=' ', skipinitialspace=True)

#Create data frames in pandas
df_lin = pd.DataFrame(data_lin)

"""
Indices of DataFrame are: 
Index([u'1.file', u'2.cLcL_NLO', u'3.cLsL_NLO', u'4.cLdL_NLO', u'5.cLuL_NLO',
       u'6.cLuR_NLO', u'7.cLdR_NLO', u'8.cLsR_NLO', u'9.cLcR_NLO',
       u'10.sLsL_NLO', u'11.sLdL_NLO', u'12.sLuL_NLO', u'13.sLuR_NLO',
       u'14.sLdR_NLO', u'15.sLsR_NLO', u'16.sLcR_NLO', u'17.dLdL_NLO',
       u'18.dLuL_NLO', u'19.dLuR_NLO', u'20.dLdR_NLO', u'21.dLsR_NLO',
       u'22.dLcR_NLO', u'23.uLuL_NLO', u'24.uLuR_NLO', u'25.uLdR_NLO',
       u'26.uLsR_NLO', u'27.uLcR_NLO', u'28.uRuR_NLO', u'29.uRdR_NLO',
       u'30.uRsR_NLO', u'31.uRcR_NLO', u'32.dRdR_NLO', u'33.dRsR_NLO',
       u'34.dRcR_NLO', u'35.sRsR_NLO', u'36.sRcR_NLO', u'37.cRcR_NLO',
       u'38.mGluino', u'39.mcL', u'40.mdL', u'41.mdR', u'42.muL', u'43.msL',
       u'44.muR', u'45.msR', u'46.mcR', u'Unnamed: 46'],
"""

#Drop BS column
df_lin = df_lin.drop('Unnamed: 46', axis=1)
mask = df_lin[target_list] != 0
df_lin = df_lin[mask]

# Lower limit
mask_lower = df_lin[target_list] > 1e-16
df_lin = df_lin[mask_lower]


#Check properties of data
print "len(data) = ", len(df_lin)
print "data.shape = ", df_lin.shape

# Define features to train on and target nubers (used below)
# Convert to array for scikit using values
# ravel is used to avoid [[n]], i.e. n x 1 arrays
mean_list = [ "39.mcL", "40.mdL", "41.mdR", "42.muL", "43.msL", "44.muR", "45.msR", "46.mcR"]
means = df_lin[mean_list].mean(axis=1).values.ravel()

cross_list = ['2.cLcL_NLO', '3.cLsL_NLO', '4.cLdL_NLO', '5.cLuL_NLO',
              '6.cLuR_NLO', '7.cLdR_NLO', '8.cLsR_NLO','9.cLcR_NLO',
       '10.sLsL_NLO','11.sLdL_NLO', '12.sLuL_NLO', '13.sLuR_NLO',
       '14.sLdR_NLO', '15.sLsR_NLO', '16.sLcR_NLO', '17.dLdL_NLO',
       '18.dLuL_NLO', '19.dLuR_NLO', '20.dLdR_NLO', '21.dLsR_NLO',
       '22.dLcR_NLO', '23.uLuL_NLO', '24.uLuR_NLO', '25.uLdR_NLO',
      '26.uLsR_NLO', '27.uLcR_NLO', '28.uRuR_NLO', '29.uRdR_NLO',
      '30.uRsR_NLO', '31.uRcR_NLO', '32.dRdR_NLO', '33.dRsR_NLO',
       '34.dRcR_NLO', '35.sRsR_NLO', '36.sRcR_NLO', '37.cRcR_NLO']

crossections = df_lin[cross_list]
crossections = crossections.sum(axis=1)



n_features = len(feature_list)
features = df_lin[feature_list].values
features_mean = np.zeros(len(features))

features_mean = (features[:,1] + features[:,2])/2.

target = df_lin[target_list].values.ravel()

long_features = np.zeros(( len(features), len(features[0])+1 ))

if len(long_features[0]) == 3:
    long_features[:,0] = features[:,0]
    long_features[:,1] = features[:,1]
    long_features[:,2] = means[:,]
    
elif len(long_features[0]) == 4:
    long_features[:,0] = features[:,0]
    long_features[:,1] = features[:,1]
    long_features[:,2] = features[:,2]
    long_features[:,3] = means[:,]


#long_features[:,0] = features[:,0]
#long_features[:,1] = features_mean
#long_features[:,2] = means[:,]

print "Max target: ", max(np.log10(target))
print "Min target: ", min(np.log10(target))

target_m2 = target/features[:,0]**2

##############################################################
# Distributed Gaussian Processes                             #
##############################################################

# Define kernel to be used in GP
if len(long_features[0]) == 3:
    kernel_matern1 = C(10, (1e-3, 100)) * Matern(np.array([10, 10, 10]), (1e-4, 1e6), nu=1.0)
elif len(long_features[0]) == 4:
    kernel_matern1 = C(10, (1e-3, 100)) * Matern(np.array([1e+4, 1e+4, 1e+4, 1000]), (1e2, 1e6), nu=1.0)

kernel_matern = C(10, (1e-3, 1000)) * Matern(np.array([100, 100, 100]), (10, 1e6), nu=1.5)+WhiteKernel(1e-5, (1e-10,1e-5) )
    
my_dgp = dgp_parallel(n_experts, outfile, kernel=kernel_matern, verbose=False, njobs=my_njobs)
my_dgp.fit_and_predict(long_features, target_m2, trainsize=trainsize)#, alpha=7.544e-07)

