"""
@author: 
    David J. Cox, PhD, MSB, BCBA-D
    dcox33@jhmi.edu
    https://www.researchgate.net/profile/David_Cox26
"""

# Packages!!!! 

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, ward

#Show current working directory. 
dirpath = os.getcwd()
print(dirpath)

#Set current working directory to the folder that contains your data.
# Home PC
os.chdir('C:/Users/coxda/Dropbox/Projects/Current Project Manuscripts/Empirical/Quant & ML for Types of Discounting/Data')
# Work Mac
os.chdir('/Users/dcox/Dropbox/Projects/Current Project Manuscripts/Empirical/Quant & ML for Types of Discounting/Data')

# Import data. 
data_all = pd.read_csv('ALLDiscountRates.csv', low_memory=False) #All data.
#data_cocaine = pd.read_csv('DiscountRatesCocaine.csv', low_memory=False) #Cocaine data.
#data_control = pd.read_csv('DiscountRatesControl.csv', low_memory=False) #Control data. 
data_cocaine_ml = pd.read_csv('DiscountRatesCocaineML.csv', low_memory=False) #Cocaine data.
data_control_ml = pd.read_csv('DiscountRatesControlML.csv', low_memory=False) #Control data. 

#Check the headers to make sure everything looks right. 
data_all.head()
data_cocaine_ml.head()
data_control_ml.head()

#Drop task names from first column for analysis. 
data_cocaine_ready = data_cocaine_ml.drop(['Unnamed: 0'], axis=1)
data_cocaine_ready.head()

data_control_ready = data_control_ml.drop(['Unnamed: 0'], axis=1)
data_control_ready.head()

#Check the shape of the data
print(data_all.shape)
print(data_cocaine.shape)
print(data_control.shape)

#Check to make sure we have all participants accounted for in each group. 
print(len(data_all))
print(len(data_cocaine))
print(len(data_control))


# Create pairplot. Shows all discounting rates as a function of all other discounting rates. 
sns.pairplot(data_cocaine)
sns.pairplot(data_control)


""""""""" 
    Identify optimal number of clusters for cocaine group from Rousseeuw, P.J. (1987). 
    Silhouettes: a Graphical Aid to the Interpretation and Validation of Cluster Analysis. 
    Computational and Applied Mathematics 20: 53–65. doi:10.1016/0377-0427(87)90125-7.
"""""""""
# Identify optimal number of clusters for cocaine group. 
Silhouette_Score_cocaine = []
agg_cocaine = range(2, 20)
for k in agg_cocaine:
    ac = AgglomerativeClustering(n_clusters = k).fit(data_cocaine_ready)
    labels = ac.labels_
    silhouette = metrics.silhouette_score(data_cocaine_ready, labels, metric='euclidean')
    Silhouette_Score_cocaine.append(silhouette)

# Identify optimal number of clusters for control group. 
Silhouette_Score_control = []
agg_control = range(2, 16)
for k in agg_control:
    ac = AgglomerativeClustering(n_clusters = k).fit(data_control_ready)
    labels = ac.labels_
    silhouette = metrics.silhouette_score(data_control_ready, labels, metric='euclidean')
    Silhouette_Score_control.append(silhouette)

# Show cluster scores. 
list(Silhouette_Score_cocaine)
list(Silhouette_Score_control)

# Create plot of silhouette scores for cocaine group
plt.plot(agg_cocaine, Silhouette_Score_cocaine, 'o-', color='black')
plt.xlabel('Number of Clusters')
plt.xticks(np.arange(0, 20, step=2))
plt.yticks(np.arange(0, 0.6, step=0.1))
plt.ylabel('Silhouette Score')
plt.title('Cocaine Group')

# Create plot of silhouette scores for control group
plt.plot(agg_control, Silhouette_Score_control, 'o-', color='black')
plt.xlabel('Number of Clusters')
plt.xticks(np.arange(0, 20, step=2))
plt.yticks(np.arange(0, 0.6, step=0.1))
plt.ylabel('Silhouette Score')
plt.title('Control Group')


"""""""""""""""""""""""""""
    Create dendograms to show agglomerative hierarchical clustering. 
"""""""""""""""""""""""""""
# Apply ward clustering to data array for cocaine group. 
linkage_array_cocaine = ward(data_cocaine_ready)

#Plot dendrogram for the linkage_array_cocaine containing distances between clusters. 
tasks_cocaine = []
for name in data_cocaine_ml['Unnamed: 0']:
    print(name)
    tasks_cocaine.append(name)

dendrogram(linkage_array_cocaine, labels=tasks_cocaine)
plt.ylabel("Cluster Distance")
plt.title("Cocaine Group")
plt.xticks(rotation=90)
ax_control = plt.gca()
bounds = ax_control.get_xbound()
ax_control.plot(bounds, [21.18, 21.18], '--', c='k')


# Apply ward clustering to data array for control group. 
linkage_array_control = ward(data_control_ready)

# Plot dendrogram for the linkage_array_control containing distances between clusters. 
tasks_control= []
for name in data_control_ml['Unnamed: 0']:
    print(name)
    tasks_control.append(name)
    
dendrogram(linkage_array_control, labels=tasks_control,)
plt.ylabel("Cluster Distance")
plt.title("Control Group")
plt.xticks(rotation=90)
ax_control = plt.gca()
bounds = ax_control.get_xbound()
ax_control.plot(bounds, [22.5, 22.5], '--', c='k')