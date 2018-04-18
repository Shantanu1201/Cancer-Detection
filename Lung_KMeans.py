# loading libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
# loading dataset
dataset = pd.read_csv('Lung_Cancer_nums.csv')

x = dataset.iloc[:, 1:10].values
x_norm = preprocessing.MinMaxScaler()
x_minmax = x_norm.fit_transform(x)

# elbow method
from sklearn.cluster import KMeans
d = []
data = scale(x)
reduced_data = PCA(n_components=2).fit_transform(data)
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state =0)
    kmeans.fit(x_minmax)
    d.append(kmeans.inertia_)
 
plt.plot(range(1, 11), d)
plt.title('Optimal no. of Clusters')
plt.xlabel('No. of clusters')
plt.ylabel('SS val')
plt.show()

for j in range(5,1,-1):
# we require 2 clusters as per graph
    kmeans = KMeans(n_clusters=j, random_state =0)

# prediticting kmeans
    y_kmeans = kmeans.fit_predict(reduced_data)

    ytest = dataset.iloc[:, 11].values
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(ytest, y_kmeans)

    Accuracy = ((cm[0][0] + cm[1][1]) / cm.sum()) *100
    print('Accuracy[%d] :'%j, Accuracy)
print ("\n")

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print(centroids)

print("\n")
print((cm[0][0]/cm.sum())*100)
print((cm[1][1]/cm.sum())*100)

colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']
for j in range(2):
    plt.scatter(reduced_data[:,0][labels == j],
                reduced_data[:,1][labels == j], marker='o', c=colors[j%10], s=3, label='class {0}'.format(j+1))

for pt in centroids:
    plt.scatter(pt[0], pt[1], marker='X', s=100, c='k')
    plt.axis('off')
    plt.title('KMean Clustering \n with Centers = 2')
    plt.savefig('Lung_KMeans.png', format='png', dpi=700)
#print ("\n")
#print(((cm[0][0]+cm[1][0])/cm.sum())*100)
#print(((cm[1][1]+cm[0][1])/cm.sum())*100)


"""
colors = ["g.","r."]
for i in range(len(x)): 
    plt.plot(x[i][0], x[i][1], colors[labels[i]], markersize = 5)

plt.scatter(centroids[:, 0],centroids[:, 1], marker = "X", s=100, linewidths = .6, zorder = 100)
plt.show()
"""
#   print("coordinate:",x[i], "label:", labels[i])
