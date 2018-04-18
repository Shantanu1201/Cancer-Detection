import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
import skfuzzy as fuzz

# loading dataset
dataset = pd.read_csv('Genome_nums.csv')

x = dataset.iloc[:, 0:17].values
x_norm = preprocessing.MinMaxScaler()
x_minmax = x_norm.fit_transform(x)

d = []
data = scale(x)
reduced_data = PCA(n_components=2).fit_transform(data)

alldata = np.vstack((reduced_data[:,0], reduced_data[:,1]))

fpcs = []
colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']

# Fuzzy C Means
for i in range(2,10):
    centroids, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(alldata, i, 2, error=0.175, maxiter=300, init=None)
    fpcs.append(fpc)
    
plt.scatter(np.r_[2:10], fpcs)
plt.xlabel('No. of clusters')
plt.ylabel('Fuzzy partition coefficient')
plt.show()

centroids, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(alldata, 2, 2, error=0.175, maxiter=300, init=None)
fpcs.append(fpc)

ytest = dataset.iloc[:, 18].values
cluster_membership = np.argmax(u, axis=0)

cm = confusion_matrix(ytest, cluster_membership)
print(cm)
print ("\n")

Accuracy = ((cm[0][1] + cm[1][0]) / cm.sum()) *100
print(100-Accuracy)
print("\n")

print(centroids)
print("\n")
print(fpc)

for j in range(2):
    plt.scatter(reduced_data[:,0][cluster_membership == j],
                reduced_data[:,1][cluster_membership == j], marker='o', c=colors[j%10], s=3, label='class {0}'.format(j+1))

for pt in centroids:
    plt.scatter(pt[0], pt[1], marker='X', s=100, c='k')

#plt.axis('off')
plt.title('Fuzzy C-Mean Clustering \n with Centers = {0}; FPC = {1:.2f}'.format(2, fpc))
plt.savefig('genome_cancer.png', format='png', dpi=700)
plt.show()
