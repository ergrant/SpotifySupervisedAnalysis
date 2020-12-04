# we are doing outlier detection with a single library
import os
import sys
import numpy as np
from numpy import random
import matplotlib
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd

print(__doc__)

np.random.seed(42)

desired_attributes = ['title', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'liveness', 'tempo']
files = {}

directory = "/Users/eringrant/github/SpotifySupervisedAnalysis/Data"
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        files[os.path.join(directory, filename)] = 0
    else:
        continue

for name in files:
    file = open(name, "r")
    files[name] = pd.read_csv(file, delimiter=',', index_col=0)

# Person we are choosing
library_to_study = random.choice(list(files))

xx, yy = np.meshgrid(np.linspace([0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1], 500))
# Convert to numpy
X = files[library_to_study].to_numpy()
np.random.shuffle(X)

X_labels, X = np.split(X, [1], axis=1)
X_normed = X / [1,1,-60,1,1,1,250]
# Generate normal (not abnormal) training observations
X_train, X_test = np.split(X_normed, [int(len(X)*0.8)])

# Generate some abnormal novel observations
#X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

# fit the model for novelty detection (novelty=True)
clf = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1)
clf.fit(X_train)
# DO NOT use predict, decision_function and score_samples on X_train as this
# would give wrong results but only on new unseen data (not used in X_train),
# e.g. X_test, X_outliers or the meshgrid
y_pred_test = clf.predict(X_test)
#y_pred_outliers = clf.predict(X_outliers)
n_error_test = y_pred_test[y_pred_test == -1].size
#n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

# plot the learned frontier, the points, and the nearest vectors to the plane
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("Novelty Detection with LOF")
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

s = 40
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s,
                 edgecolors='k')
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s,
                edgecolors='k')
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([a.collections[0], b1, b2, c],
           ["learned frontier", "training observations",
            "new regular observations", "new abnormal observations"],
           loc="upper left",
           prop=matplotlib.font_manager.FontProperties(size=11))
plt.xlabel(
    "errors novel regular: %d/40 ; errors novel abnormal: %d/40"
    % (n_error_test, n_error_outliers))
plt.show()
