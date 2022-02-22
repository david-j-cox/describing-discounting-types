"""
@author: 
    David J. Cox, PhD, MSB, BCBA-D
    dcox33@jhmi.edu
    https://www.researchgate.net/profile/David_Cox26
"""

# Packages!!!! 
import os
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


#Show current working directory. 
dirpath = os.getcwd()
print(dirpath)

#Set current working directory to the folder that contains your data.
# Home PC
os.chdir('C:/Users/coxda/Dropbox/Projects/Current Project Manuscripts/Empirical/Are there different types of discounting/Data')
# Work Mac
os.chdir('/Users/dcox/Dropbox/Projects/Current Project Manuscripts/Empirical/Are there different types of discounting/Data')


# Import data. 
data_all = pd.read_csv('ALLDiscountRates.csv', low_memory=False) #All data.
data_cocaine = pd.read_csv('DiscountRatesCocaine.csv', low_memory=False) #Cocaine data.
data_control = pd.read_csv('DiscountRatesControl.csv', low_memory=False) #Control data. 
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

"""""""""""""""""""""""""""
    Commence DBSCAN. 
"""""""""""""""""""""""""""
# Identify optimal number of clusters for cocaine group. 
Silhouette_Score_cocaine=[]
unique_categories_cocaine=[]
category_labels_cocaine = []
db_cocaine = range(5, 37)
for i in db_cocaine:
    dbscan = DBSCAN(eps = i, min_samples=1)
    clusters = dbscan.fit(data_cocaine_ready)
    labels = clusters.labels_
    silhouette = metrics.silhouette_score(data_cocaine_ready, labels, metric='euclidean')
    Silhouette_Score_cocaine.append(silhouette)
    cluster_counts=np.unique(labels)
    cluster_add=max(cluster_counts)+1
    unique_categories_cocaine.append(cluster_add)
    category_labels_cocaine.append(labels)

# Identify optimal number of clusters for cocaine group. 
Silhouette_Score_control=[]
unique_categories_control = []
category_labels_control = []
db_control = range(7, 28)
for i in db_control:
    dbscan = DBSCAN(eps = i, min_samples=1)
    clusters = dbscan.fit(data_control_ready)
    labels = clusters.labels_
    silhouette = metrics.silhouette_score(data_control_ready, labels, metric='euclidean')
    Silhouette_Score_control.append(silhouette)
    cluster_counts=np.unique(labels)
    cluster_add=max(cluster_counts)+1
    unique_categories_control.append(cluster_add)
    category_labels_control.append(labels)

# Show cluster scores. 
list(Silhouette_Score_cocaine)
list(Silhouette_Score_control)

# Show category labels
list(category_labels_cocaine)
list(category_labels_control)

# Show categories at each ep. 
list(unique_categories_cocaine)
list(unique_categories_control)

# Create plot of silhouette scores for cocaine group as function of eps. 
plt.plot(db_cocaine, Silhouette_Score_cocaine, 'o-', color='black')
plt.xlabel('Number of Eps')
plt.xticks(np.arange(0, 40, step=2))
plt.yticks(np.arange(0, 0.6, step=0.1))
plt.ylabel('Silhouette Score')
plt.title('Cocaine Group')

# Create plot of silhouette scores for control group as function of eps. 
plt.plot(db_control, Silhouette_Score_control, 'o-', color='black')
plt.xlabel('Number of Eps')
plt.xticks(np.arange(0, 40, step=2))
plt.yticks(np.arange(0, 0.6, step=0.1))
plt.ylabel('Silhouette Score')
plt.title('Control Group')   
plt.grid(True) 


# Create plot of silhouette for cocaine group as function of cluster numbers. 
plt.plot(unique_categories_cocaine, Silhouette_Score_cocaine, 'o-', color='black')
plt.xlabel('Number of Clusters')
plt.xticks(np.arange(0, 20, step=2))
plt.yticks(np.arange(0, 0.6, step=0.1))
plt.ylabel('Silhouette Score')
plt.title('Cocaine Group')


# Create plot of silhouette for control group as function of cluster numbers. 
plt.plot(unique_categories_control, Silhouette_Score_control, 'o-', color='black')
plt.xlabel('Number of Clusters')
plt.xticks(np.arange(0, 20, step=2))
plt.yticks(np.arange(0, 0.6, step=0.1))
plt.ylabel('Silhouette Score')
plt.title('Control Group')
plt.grid(True)


# Create plot of clusters for cocaine group as function of eps. 
plt.plot(db_cocaine,unique_categories_cocaine, 'o-', color='black')
plt.xlabel('Eps')
plt.xticks(np.arange(0, 38, step=2))
plt.yticks(np.arange(0, 20, step=2))
plt.ylabel('Number of Clusters')
plt.title('Cocaine Group')
plt.grid(True)

# Create plot of clusters for control group as function of eps. 
plt.plot(db_control, unique_categories_control,'o-', color='black')
plt.xlabel('Eps')
plt.xticks(np.arange(0, 38, step=2))
plt.yticks(np.arange(0, 20, step=2))
plt.ylabel('Number of Clusters')
plt.title('Control Group')
plt.grid(True)