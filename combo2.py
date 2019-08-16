#!/usr/bin/python
import random
#from udacityplots import *
import warnings
warnings.filterwarnings("ignore")

import matplotlib 
matplotlib.use('agg')

import matplotlib.pyplot as plt
import pylab as pl
import numpy as np


###############################################################################
### make the toy dataset
n_points = 1000
random.seed(42)
grade = [random.random() for ii in range(0,n_points)]
bumpy = [random.random() for ii in range(0,n_points)]
error = [random.random() for ii in range(0,n_points)]
y = [round(grade[ii]*bumpy[ii]+0.3+0.1*error[ii]) for ii in range(0,n_points)]
for ii in range(0, len(y)):
    if grade[ii]>0.8 or bumpy[ii]>0.8:
        y[ii] = 1.0

### split into train/test sets
X = [[gg, ss] for gg, ss in zip(grade, bumpy)]
split = int(0.75*n_points)
X_train = X[0:split]
X_test  = X[split:]
y_train = y[0:split]
y_test  = y[split:]

grade_sig = [X_train[ii][0] for ii in range(0, len(X_train)) if y_train[ii]==0]
bumpy_sig = [X_train[ii][1] for ii in range(0, len(X_train)) if y_train[ii]==0]
grade_bkg = [X_train[ii][0] for ii in range(0, len(X_train)) if y_train[ii]==1]
bumpy_bkg = [X_train[ii][1] for ii in range(0, len(X_train)) if y_train[ii]==1]

#    training_data = {"fast":{"grade":grade_sig, "bumpiness":bumpy_sig}
#            , "slow":{"grade":grade_bkg, "bumpiness":bumpy_bkg}}


grade_sig = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==0]
bumpy_sig = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==0]
grade_bkg = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==1]
bumpy_bkg = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==1]

test_data = {"fast":{"grade":grade_sig, "bumpiness":bumpy_sig}, "slow":{"grade":grade_bkg, "bumpiness":bumpy_bkg}}



############################################################################################################
from sklearn.naive_bayes import GaussianNB as GNB

x_min = 0.0; x_max = 1.0
y_min = 0.0; y_max = 1.0

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].

clf = GNB()
clf.fit(X_train, y_train)

pred = clf.predict(X_test)

h = .01  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.pcolormesh(xx, yy, Z, cmap=pl.cm.seismic)

# Plot also the test points
grade_sig = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==0]
bumpy_sig = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==0]
grade_bkg = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==1]
bumpy_bkg = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==1]

plt.scatter(grade_sig, bumpy_sig, color = "b", label="fast")
plt.scatter(grade_bkg, bumpy_bkg, color = "r", label="slow")
plt.show()
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.savefig("test.png")
