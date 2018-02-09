"""
For plotting the results of Distributed Gaussian Processes, 
calulated using the class dgp.py.

@author: Ingrid A V Holm
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import pandas as pd
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
    matern = True
else:
    matern = False

numbers = []

################################################
# Read infile using Pandas                     #
################################################

df_lin = pd.read_csv(infile, sep=" ", header=None, skiprows=1, skipinitialspace=True)
df_lin.columns = ["Error", "Mus", "Sigmas", "Y_test", "Mg", "Mq", "Unnamed: 4"]
df_lin = df_lin.drop(["Unnamed: 4"], axis=1)

# Change back to cross section for sigma_m2
if matern == True:
    df_lin[["Y_test", "Mus"]] = df_lin[["Y_test", "Mus"]].add(2*np.log10(df_lin["Mg"]), axis="index")

errors = df_lin["Error"].values.ravel()
mus = df_lin["Mus"].values.ravel()
y_test = df_lin["Y_test"].values.ravel()
mg = df_lin["Mg"].values.ravel()
msq = df_lin["Mq"].values.ravel()

N = len(y_test)

"""
titles = ["/home/ingrid/Documents/Master/ML/Final_remarks/Matern110_nozeros/feature_dist_lin_mat44m2.pdf",\
          "/home/ingrid/Documents/Master/ML/Final_remarks/Matern110_nozeros/hist_mat44m2.pdf",\
          "/home/ingrid/Documents/Master/ML/Final_remarks/Matern110_nozeros/sigmamq_true_lin_mat44m2.pdf",\
          "/home/ingrid/Documents/Master/ML/Final_remarks/Matern110_nozeros/sigmamq_predicted_lin_mat44m2.pdf",\
          "/home/ingrid/Documents/Master/ML/Final_remarks/Matern110_nozeros/sigmamg_true_lin_mat44m2.pdf",\
          "/home/ingrid/Documents/Master/ML/Final_remarks/Matern110_nozeros/sigmamg_predicted_lin_mat44m2.pdf",\
          "/home/ingrid/Documents/Master/ML/Final_remarks/Matern110_nozeros/sigmamq_true_overunder_mat44m2.pdf",\
          "/home/ingrid/Documents/Master/ML/Final_remarks/Matern110_nozeros/sigmamq_predicted_overunder_mat44m2.pdf"
]
"""

##############################################
# Plot features                              #
##############################################

k = plt.figure(10)
k.subplotpars.update(left=0.12)
plt.scatter(mg, msq)
plt.title('Feature distribution (data quality)')
plt.xlabel(r'$m_{\tilde{g}}$')
plt.ylabel(r'$m_{\tilde{q}}$')
# # # # # # # # # # # # #  # # # # # # 
plt.savefig(titles[0])
# # # # # # # # # # # # #  # # # # # #
k.show()
#plt.show()

# Choose limits for Gaussian fits
plotlims = 1.0
lims1 = 0.5
lims2 = 0.25
n_bins_better = int(2*plotlims*25)

# Find histograms within limits

errors_new_1 = df_lin.loc[abs(df_lin["Error"]) <= lims1]["Error"].values.ravel()
errors_new_2 = df_lin.loc[abs(df_lin["Error"]) <= lims2]["Error"].values.ravel()
errors_plot = df_lin.loc[abs(df_lin["Error"]) <= plotlims]["Error"].values.ravel()

per_1 = 1 - len(errors_new_1)/float(N)
per_2 = 1 - len(errors_new_2)/float(N)
per_plot = 1 - len(errors_plot)/float(N)

#Print to screen the number of points outside desired interval

print "DGP: The fraction of points outside (- %.2f, %.2f ) is %.4f" % (lims1, lims1, per_1)
print "DGP: The fraction of points outside (- %.2f, %.2f ) is %.4f" % (lims2, lims2, per_2)
print "DGP: The fraction of points outside (- %.2f, %.2f ) is %.4f" % (plotlims, plotlims, per_plot)
print "New number of test points is  %.f " % len(errors_new_1)

# Find Gaussian approximations

(m_1, s_1) = norm.fit(errors_new_1)
(m_2, s_2) = norm.fit(errors_new_2)

####################################################
# Plot error histogram                             #
####################################################

"""
if my_kernel == 'kernel6':
    plt.title(r'$k=C_1 \exp(-x^2/\ell^2)+ C_2$')
elif my_kernel == 'kernel_ell':
    plt.title(r'$k=C_1 \exp(-(x^T M x))+ C_2$')
elif my_kernel == 'kernel_matern':
    plt.title(r'$k=C_1 \frac{2^{1-\nu}}{\Sigma(\nu)} \Bigg( \frac{\sqrt{2 \nu} r}{\ell} \Bigg)^{\nu} K_{\nu}  \Bigg( \frac{\sqrt{2 \nu} r}{\ell} \Bigg)$')
"""

n = plt.figure(9)
n.subplotpars.update(top=0.95)

plt.hist(errors_plot, normed = True, bins= n_bins_better, linewidth=2)
xt = plt.xticks()[0]
xmin, xmax = min(xt), max(xt)
lnspc = np.linspace(xmin, xmax, len(errors_plot))
pdf_1 = stats.norm.pdf(lnspc, m_1, s_1)
pdf_2 = stats.norm.pdf(lnspc, m_2, s_2)

plt.plot(lnspc, pdf_1, lnspc, pdf_2)   
plt.legend(['$\mu$: %.3f, $\sigma^2$: %.3f (0.50)' % (m_1, s_1),'$\mu$: %.3f, $\sigma^2$: %.3f (0.25)' % (m_2, s_2)], loc='upper right', fontsize='x-large')
plt.xlim(-plotlims, plotlims)
plt.xlabel(r'$\varepsilon$')
plt.ylabel('n')

# # # # # # # # # # # # #  # # # # # # 
plt.savefig(titles[1])
# # # # # # # # # # # # #  # # # # # #

n.show()


######################################################################
# Plot each prediction with error bar                                #
######################################################################

f = plt.figure(1)
f.subplotpars.update(left=0.12)
plt.scatter(msq, y_test, alpha = 0.7)
plt.title('Cross section (true)')
plt.xlabel(r'm_{\tilde{q}}')
plt.ylabel(r'$\log_{10}(\sigma/\sigma^0)$, $\sigma^0 = 1$ fb')
# # # # # # # # # # # # #  # # # # # # 
plt.savefig(titles[2])
# # # # # # # # # # # # #  # # # # # #
f.show()

h = plt.figure(2)
h.subplotpars.update(left=0.12)
plt.scatter(msq, mus, alpha=0.7, color='royalblue')
plt.title('Cross section (predicted)')
plt.xlabel(r'm_{\tilde{q}}')
plt.ylabel(r'$\log_{10}(\sigma/\sigma^0)$, $\sigma^0 = 1$ fb')
# # # # # # # # # # # # #  # # # # # # 
plt.savefig(titles[3])
# # # # # # # # # # # # #  # # # # # #
f.show()

g = plt.figure(3)
g.subplotpars.update(left=0.12)
plt.scatter(mg, y_test, alpha=0.7)
plt.title('Cross section (true)')
plt.xlabel(r'm_{\tilde{g}}')
plt.ylabel(r'$\log_{10}(\sigma/\sigma^0)$, $\sigma^0 = 1$ fb')
# # # # # # # # # # # # #  # # # # # # 
plt.savefig(titles[4])
# # # # # # # # # # # # #  # # # # # #
g.show()

p = plt.figure(4)
p.subplotpars.update(left=0.12)
plt.scatter(mg, y_test, color='royalblue', alpha=0.7)
plt.title('Cross section (predicted)')
plt.xlabel(r'm_{\tilde{g}}')
plt.ylabel(r'$\log_{10}(\sigma/\sigma^0)$, $\sigma^0 = 1$ fb')
# # # # # # # # # # # # #  # # # # # # 
plt.savefig(titles[5])
# # # # # # # # # # # # #  # # # # # #
g.show()

#######################################################
# Split into mg-areas                                 #
#######################################################

mglim = 2000

msq_under = df_lin.loc[df_lin["Mg"] <= mglim]["Mq"].values.ravel()
mus_under = df_lin.loc[df_lin["Mg"] <= mglim]["Mus"].values.ravel()
y_test_under = df_lin.loc[df_lin["Mg"] <= mglim]["Y_test"].values.ravel()

msq_over = df_lin.loc[df_lin["Mg"] >= mglim]["Mq"].values.ravel()
mus_over = df_lin.loc[df_lin["Mg"] >= mglim]["Mus"].values.ravel()
y_test_over = df_lin.loc[df_lin["Mg"] >= mglim]["Y_test"].values.ravel()

u = plt.figure(11)
u.subplotpars.update(left=0.12)
plt.scatter(msq_under, y_test_under, color='mediumvioletred')
plt.scatter(msq_over, y_test_over)
plt.title('Cross section (true, lin)')
plt.xlabel(r'm_{\tilde{q}}')
plt.ylabel(r'$\log_{10}(\sigma/\sigma^0)$, $\sigma^0 = 1$ fb')
plt.legend([r'$m_{\tilde{g}} < 2000$',r'$m_{\tilde{g}} > 2000$'],fontsize='x-large')
# # # # # # # # # # # # #  # # # # # # 
plt.savefig(titles[6])
# # # # # # # # # # # # #  # # # # # #
u.show()

o = plt.figure(12)
o.subplotpars.update(left=0.12)
plt.scatter(msq_under, mus_under, alpha=0.7, color='royalblue')
plt.scatter(msq_over, mus_over, alpha=0.7, color='lightblue')
plt.title('Cross section (predicted, lin)')
plt.xlabel(r'm_{\tilde{q}}')
plt.ylabel(r'$\log_{10}(\sigma/\sigma^0)$, $\sigma^0 = 1$ fb')
plt.legend([r'$m_{\tilde{g}} < 2000$',r'$m_{\tilde{g}} > 2000$'], fontsize='x-large')
# # # # # # # # # # # # #  # # # # # # 
plt.savefig(titles[7])
# # # # # # # # # # # # #  # # # # # #
o.show()

plt.show()
