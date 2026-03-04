# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 10:45:33 2023

@author: Logan collier
"""

# synthetic classification dataset
from numpy import where
from matplotlib import pyplot
import numpy as np
import glob, os
import random
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA #Principal Component Analysis
from sklearn.manifold import TSNE #T-Distributed Stochastic Neighbor Embedding
from sklearn.cluster import KMeans #K-Means Clustering
from sklearn.preprocessing import StandardScaler #used for 'Feature Scaling'
import pandas as pd

#plotly imports
import plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
'''

'''
def dataload(classFolder):
    cnt = 0
    class_labels = []
    class_samples = []
    for file in os.listdir(classFolder):
        
        data = np.load(classFolder+'//'+file)
        print(file)
        for i in data.keys():
            for j in data.get(i):
                class_samples.append(j)
                class_labels.append(file.strip(".npz"))
        if cnt == 2:
            break
        cnt+=1
    return class_labels, class_samples
                
def shuffle(class_labels, class_samples):
    temp = list(zip(class_labels, class_samples))
    random.shuffle(temp)
    res1, res2 = zip(*temp)
    # res1 and res2 come out as tuples, and so must be converted to lists.
    res1, res2 = list(res1), list(res2)
    return res1, res2

class_labels, class_samples = dataload(os.getcwd()+'\\extract')
print(class_samples[0].shape)
#print(class_labels)
print(len(class_samples))
print(len(class_labels))
class_colors = set()
for i in class_labels:
    class_colors.add(i)
rgb_values = sns.color_palette("Set2", 4)
color_map = dict(zip(class_colors, rgb_values))

#class_labels, class_samples = shuffle(class_labels,class_samples)

class_samples = np.asarray(class_samples,dtype='float32')
class_labels = np.asarray(class_labels)
#class_samples = np.expand_dims(class_samples, axis=-1)

nsamples, nx, ny = class_samples.shape
d2_train_dataset = class_samples.reshape((nsamples,nx*ny))
#d2_train_labels = class_labels.reshape((nsamples,nx*ny))
df =pd.DataFrame(np.rot90(d2_train_dataset))
df['class'] = class_labels
print(df.head())
plt.scatter(df[0],df[1],c=df['class'].map(color_map))








#pca = PCA(n_components=100, random_state=22)
#pca.fit(d2_train_dataset)
#x = pca.transform(d2_train_dataset)
#X = pd.DataFrame(d2_train_dataset)
kmeans = KMeans(n_clusters=6)

kmeans.fit(d2_train_dataset)
clusters = kmeans.predict(np.array(d2_train_dataset))
print(clusters)
centroids = kmeans.cluster_centers_
print(centroids)
print(centroids[0])
plt.scatter(d2_train_dataset[ : , 0], d2_train_dataset[ : , 1], s =50, c='b')
plt.scatter(centroids[0][0], centroids[0][1], s=200, c='g', marker='s')
plt.scatter(centroids[1][0], centroids[1][1], s=200, c='r', marker='s')
plt.show()

d2_train_dataset["Cluster"] = clusters


plotX = pd.DataFrame(d2_train_dataset)
#PCA with one principal component
pca_1d = PCA(n_components=1)

#PCA with two principal components
pca_2d = PCA(n_components=2)

#PCA with three principal components
pca_3d = PCA(n_components=3)
#This DataFrame holds that single principal component mentioned above
PCs_1d = pd.DataFrame(pca_1d.fit_transform(plotX.drop(["Cluster"], axis=1)))

#This DataFrame contains the two principal components that will be used
#for the 2-D visualization mentioned above
PCs_2d = pd.DataFrame(pca_2d.fit_transform(plotX.drop(["Cluster"], axis=1)))

#And this DataFrame contains three principal components that will aid us
#in visualizing our clusters in 3-D
PCs_3d = pd.DataFrame(pca_3d.fit_transform(plotX.drop(["Cluster"], axis=1)))

PCs_1d.columns = ["PC1_1d"]

#"PC1_2d" means: 'The first principal component of the components created for 2-D visualization, by PCA.'
#And "PC2_2d" means: 'The second principal component of the components created for 2-D visualization, by PCA.'
PCs_2d.columns = ["PC1_2d", "PC2_2d"]

PCs_3d.columns = ["PC1_3d", "PC2_3d", "PC3_3d"]

plotX = pd.concat([plotX,PCs_1d,PCs_2d,PCs_3d], axis=1, join='inner')

cluster0 = plotX[plotX["Cluster"] == 0]
cluster1 = plotX[plotX["Cluster"] == 1]
cluster2 = plotX[plotX["Cluster"] == 2]
#Instructions for building the 3-D plot
#trace1 is for 'Cluster 0'
trace1 = go.Scatter(
                    x = cluster0["PC1_1d"],
                    y = cluster0["dummy"],
                    mode = "markers",
                    name = "Cluster 0",
                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                    text = None)

#trace2 is for 'Cluster 1'
trace2 = go.Scatter(
                    x = cluster1["PC1_1d"],
                    y = cluster1["dummy"],
                    mode = "markers",
                    name = "Cluster 1",
                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
                    text = None)

#trace3 is for 'Cluster 2'
trace3 = go.Scatter(
                    x = cluster2["PC1_1d"],
                    y = cluster2["dummy"],
                    mode = "markers",
                    name = "Cluster 2",
                    marker = dict(color = 'rgba(0, 255, 200, 0.8)'),
                    text = None)

data = [trace1, trace2, trace3]

title = "Visualizing Clusters in One Dimension Using PCA"

layout = dict(title = title,
              xaxis= dict(title= 'PC1',ticklen= 5,zeroline= False),
              yaxis= dict(title= '',ticklen= 5,zeroline= False)
             )

fig = dict(data = data, layout = layout)

iplot(fig)
'''

num_samples_total = 1000
cluster_centers = [(3,3), (7,7)]
num_classes = 6
epsilon = 1.0
min_samples = 13
db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(class_samples)

no_clusters = len(np.unique(class_labels) )
no_noise = np.sum(np.array(class_labels) == -1, axis=0)

print('Estimated no. of clusters: %d' % no_clusters)
print('Estimated no. of noise points: %d' % no_noise)
'''

'''

for i in data.keys():
    for j in data.get(i):
        x_train.append(j)
print(len(x_train))
#data = np.asmatrix(data,np.float())
train = np.asarray(x_train,dtype='float32')
split = 60
x_train = train#[:split]
x_test = train[split:]
print(len(x_train))
print(x_train.shape)
dim1 = x_train.shape[2]
dim2 = x_train.shape[1]
'''