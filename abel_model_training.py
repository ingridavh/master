from dgp_parallel import dgp_parallel
import sys
import numpy as np
import sklearn as sk
from sklearn.gaussian_process.kernels import RationalQuadratic, RBF, WhiteKernel, ExpSineSquared, DotProduct, Matern, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.cross_validation import train_test_split
import pandas as pd

from sklearn.externals import joblib

# Set training size
if len(sys.argv) >= 2:
    trainsize = float(sys.argv[1])
    if trainsize >= 1:
        trainsize = int(trainsize)
else:
    trainsize = 0.001

if len(sys.argv) >= 3:
    process = str(sys.argv[2])
else:
    print "Please provide a process"
    sys.exit(0)

#################################################################
# Read in and treat data (mostly) using pandas.                 #
#################################################################

# Read in data (last column is BS)
data_lin = pd.read_csv('/usit/abel/u1/ingraho/combined_files_all/data_290k.dat', sep=' ', skipinitialspace=True)

#Create data frames in pandas
df_lin = pd.DataFrame(data_lin)

# UL-processes

if process == 'uLuL':
    target_list = ["23.uLuL_NLO"]
    feature_list = ["38.mGluino", "42.muL"]
elif process == 'uLdL':
    target_list = ['18.dLuL_NLO']
    feature_list = ['38.mGluino', '42.muL', '40.mdL']
elif process == 'uLsL':
    target_list = ['12.sLuL_NLO']
    feature_list = ['38.mGluino', '42.muL', '43.msL']
elif process == 'uLcL':
    target_list = ['5.cLuL_NLO']
    feature_list = ['38.mGluino', '42.muL', '39.mcL']
elif process == 'uLuR':
    target_list = ['24.uLuR_NLO']
    feature_list = ['38.mGluino', '42.muL', '44.muR']
elif process == 'uLdR':
    target_list = ['25.uLdR_NLO']
    feature_list = ['38.mGluino', '42.muL', '41.mdR']
elif process == 'uLsR':
    target_list = ['26.uLsR_NLO']
    feature_list = ['38.mGluino', '42.muL', '45.msR']
elif process == 'uLcR':
    target_list = ['27.uLcR_NLO']
    feature_list = ['38.mGluino', '42.muL', '46.mcR']

elif process == 'dLuL':
    target_list = ["18.dLuL_NLO"]
    feature_list = ["38.mGluino", "40.mdL", "42.muL"]
elif process == 'dLdL':
    target_list = ["17.dLdL_NLO"]
    feature_list = ["38.mGluino", "40.mdL"]
elif process == 'dLsL':
    target_list = ["11.sLdL_NLO"]
    feature_list = ["38.mGluino", "40.mdL", "43.msL"]
elif process == 'dLcL':
    target_list = ["4.cLdL_NLO"]
    feature_list = ["38.mGluino", "40.mdL", "39.mcL"]
elif process == 'dLuR':
    target_list = ["19.dLuR_NLO"]
    feature_list = ["38.mGluino", "40.mdL", "44.muR"]
elif process == 'dLdR':
    target_list = ["20.dLdR_NLO"]
    feature_list = ["38.mGluino", "40.mdL", "41.mdR"]
elif process == 'dLsR':
    target_list = ["21.dLsR_NLO"]
    feature_list = ["38.mGluino", "40.mdL", "45.msR"]
elif process == 'dLcR':
    target_list = ["22.dLcR_NLO"]
    feature_list = ["38.mGluino", "40.mdL", "46.mcR"]

elif process == 'sLuL':
    target_list = ["12.sLuL_NLO"]
    feature_list = ["38.mGluino", "43.msL", "42.muL"]
elif process == 'sLdL':
    target_list = ["11.sLdL_NLO"]
    feature_list = ["38.mGluino", "43.msL", "40.mdL"]    
elif process == 'sLsL':
    target_list = ["10.sLsL_NLO"]
    feature_list = ["38.mGluino", "43.msL"]
elif process == 'sLcL':
    target_list = ["3.cLsL_NLO"]
    feature_list = ["38.mGluino", "43.msL", "39.mcL"]
elif process == 'sLuR':
    target_list = ["13.sLuR_NLO"]
    feature_list = ["38.mGluino", "43.msL", "44.muR"]
elif process == 'sLdR':
    target_list = ["14.sLdR_NLO"]
    feature_list = ["38.mGluino", "43.msL", "41.mdR"]
elif process == 'sLsR':
    target_list = ["15.sLsR_NLO"]
    feature_list = ["38.mGluino", "43.msL", "45.msR"]
elif process == 'sLcR':
    target_list = ["16.sLcR_NLO"]
    feature_list = ["38.mGluino", "43.msL", "46.mcR"]

elif process == 'cLuL':
    target_list = ["5.cLuL_NLO"]
    feature_list = ["38.mGluino", "39.mcL", "42.muL"]
elif process == 'cLdL':
    target_list = ["4.cLdL_NLO"]
    feature_list = ["38.mGluino", "39.mcL", "40.mdL"]
elif process == 'cLsL':
    target_list = ["3.cLsL_NLO"]
    feature_list = ["38.mGluino", "39.mcL", "43.msL"]
elif process == 'cLcL':
    target_list = ["2.cLcL_NLO"]
    feature_list = ["38.mGluino", "39.mcL"]
elif process == 'cLuR':
    target_list = ["6.cLuR_NLO"]
    feature_list = ["38.mGluino", "39.mcL", "44.muR"]
elif process == 'cLdR':
    target_list = ["7.cLdR_NLO"]
    feature_list = ["38.mGluino", "39.mcL", "41.mdR"]
elif process == 'cLsR':
    target_list = ["8.cLsR_NLO"]
    feature_list = ["38.mGluino", "39.mcL", "45.msR"]
elif process == 'cLcR':
    target_list = ["9.cLcR_NLO"]
    feature_list = ["38.mGluino", "39.mcL", "46.mcR"]

elif process == 'uRuL':
    target_list = ["24.uLuR_NLO"]
    feature_list = ["38.mGluino", "44.muR", "42.muL"]
elif process == 'uRdL':
    target_list = ["19.dLuR_NLO"]
    feature_list = ["38.mGluino", "44.muR", "40.mdL"]
elif process == 'uRsL':
    target_list = ["13.sLuR_NLO"]
    feature_list = ["38.mGluino", "44.muR", "43.msL"]
elif process == 'uRcL':
    target_list = ["6.cLuR_NLO"]
    feature_list = ["38.mGluino", "44.muR", "39.mcL"]
elif process == 'uRuR':
    target_list = ["28.uRuR_NLO"]
    feature_list = ["38.mGluino", "44.muR"]
elif process == 'uRdR':
    target_list = ["29.uRdR_NLO"]
    feature_list = ["38.mGluino", "44.muR", "41.mdR"]
elif process == 'uRsR':
    target_list = ["30.uRsR_NLO"]
    feature_list = ["38.mGluino", "44.muR", "45.msR"]
elif process == 'uRcR':
    target_list = ["31.uRcR_NLO"]
    feature_list = ["38.mGluino", "44.muR", "46.mcR"]

elif process == 'dRuL':
    target_list = ["25.uLdR_NLO"]
    feature_list = ["38.mGluino", "41.mdR", "42.muL"]
elif process == 'dRdL':
    target_list = ["20.dLdR_NLO"]
    feature_list = ["38.mGluino", "41.mdR", "40.mdL"]
elif process == 'dRsL':
    target_list = ["14.sLdR_NLO"]
    feature_list = ["38.mGluino", "41.mdR", "43.msL"]
elif process == 'dRcL':
    target_list = ["7.cLdR_NLO"]
    feature_list = ["38.mGluino", "41.mdR", "39.mcL"]
elif process == 'dRuR':
    target_list = ["29.uRdR_NLO"]
    feature_list = ["38.mGluino", "41.mdR", "44.muR"]
elif process == 'dRdR':
    target_list = ["32.dRdR_NLO"]
    feature_list = ["38.mGluino", "41.mdR"]
elif process == 'dRsR':
    target_list = ["33.dRsR_NLO"]
    feature_list = ["38.mGluino", "41.mdR", "45.msR"]
elif process == 'dRcR':
    target_list = ["34.dRcR_NLO"]
    feature_list = ["38.mGluino", "41.mdR", "46.mcR"]

elif process == 'sRuL':
    target_list = ["26.uLsR_NLO"]
    feature_list = ["38.mGluino", "45.msR", "42.muL"]
elif process == 'sRdL':
    target_list = ["21.dLsR_NLO"]
    feature_list = ["38.mGluino", "45.msR", "40.mdL"]
elif process == 'sRsL':
    target_list = ["15.sLsR_NLO"]
    feature_list = ["38.mGluino", "45.msR", "43.msL"]
elif process == 'sRcL':
    target_list = ["8.cLsR_NLO"]
    feature_list = ["38.mGluino", "45.msR", "39.mcL"]
elif process == 'sRuR':
    target_list = ["30.uRsR_NLO"]
    feature_list = ["38.mGluino", "45.msR", "44.muR"]
elif process == 'sRdR':
    target_list = ["33.dRsR_NLO"]
    feature_list = ["38.mGluino", "45.msR", "41.mdR"]
elif process == 'sRsR':
    target_list = ["35.sRsR_NLO"]
    feature_list = ["38.mGluino", "45.msR"]
elif process == 'sRcR':
    target_list = ["36.sRcR_NLO"]
    feature_list = ["38.mGluino", "45.msR", "46.mcR"]

elif process == 'cRcR':
    target_list = ["37.cRcR_NLO"]
    feature_list = ["38.mGluino", "46.mcR"]
    
    
#Drop BS column
df_lin = df_lin.drop('Unnamed: 46', axis=1)

mask1 = df_lin[target_list[0]] != 0
df_lin = df_lin[mask1]


# Lower limit
mask_lower = df_lin[target_list[0]] > 1e-16
df_lin = df_lin[mask_lower]
    
mean_list = ["39.mcL", "40.mdL", "41.mdR", "42.muL", "43.msL", "44.muR", "45.msR", "46.mcR"]
means = df_lin[mean_list].mean(axis=1).values.ravel()

# Define features to train on and target nubers (used below)
# Convert to array for scikit using values
# ravel is used to avoid [[n]], i.e. n x 1 arrays
features = df_lin[feature_list].values
target = df_lin[target_list].values.ravel()
target_m2 = target/features[:,0]**2

long_features = np.zeros(( len(features), len(features[0])+1 ))
long_features[:,0] = features[:,0]
long_features[:,1] = features[:,1]

if len(features[0]) == 2:
    long_features[:,2] = means[:,]
    kernel_matern1 = C(10, (1e-3, 100)) * Matern(np.array([10, 10, 10]), (1e-4, 1e6), nu=2.5)+ WhiteKernel(1, (2e-10,1e2))
    
elif len(features[0]) == 3:
    long_features[:,2] = features[:,2]
    long_features[:,3] = means[:,]
    kernel_matern1 = C(10, (1e-3, 100)) * Matern(np.array([10, 10, 10, 10]), (1e-4, 1e6), nu=2.5)+ WhiteKernel(1, (2e-10,1e2))

print "Max target: ", max(np.log10(target))
print "Min target: ", min(np.log10(target))

target_m2 = np.log10(target_m2)


X_train, X_test, y_train, y_test = train_test_split(long_features, target_m2, random_state=42, train_size = trainsize)

# Fit Gaussian process
    
gp = GaussianProcessRegressor(kernel=kernel_matern1, alpha=7.544e-07, random_state=42)
gp.fit(X_train, y_train)

print gp.kernel_
name = process+'_'+str(trainsize)+'_25_wk'
print name, type(name)
joblib.dump(gp, name)

