"""
@author: 
    David J. Cox, PhD, MSB, BCBA-D
    dcox33@jhmi.edu
    https://www.researchgate.net/profile/David_Cox26
"""

# Packages!! 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
import seaborn as sns

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

"""""""""""""""""""""""""""""""""
    Commence k-means clustering. 
"""""""""""""""""""""""""""""""""
# Build clustering models for cocaine group. 
kmeans_2 = KMeans(n_clusters =2)
kmeans_2.fit(data_cocaine_ready)

kmeans_3 = KMeans(n_clusters =3)
kmeans_3.fit(data_cocaine_ready)

kmeans_4 = KMeans(n_clusters =4)
kmeans_4.fit(data_cocaine_ready)

kmeans_5 = KMeans(n_clusters =5)
kmeans_5.fit(data_cocaine_ready)

kmeans_6 = KMeans(n_clusters =6)
kmeans_6.fit(data_cocaine_ready)

kmeans_7 = KMeans(n_clusters =7)
kmeans_7.fit(data_cocaine_ready)

kmeans_8 = KMeans(n_clusters =8)
kmeans_8.fit(data_cocaine_ready)

kmeans_9 = KMeans(n_clusters =9)
kmeans_9.fit(data_cocaine_ready)

kmeans_10 = KMeans(n_clusters =10)
kmeans_10.fit(data_cocaine_ready)

kmeans_11 = KMeans(n_clusters =11)
kmeans_11.fit(data_cocaine_ready)

kmeans_12 = KMeans(n_clusters =12)
kmeans_12.fit(data_cocaine_ready)

kmeans_13 = KMeans(n_clusters =13)
kmeans_13.fit(data_cocaine_ready)

kmeans_14 = KMeans(n_clusters =14)
kmeans_14.fit(data_cocaine_ready)

kmeans_15 = KMeans(n_clusters =15)
kmeans_15.fit(data_cocaine_ready)

kmeans_16 = KMeans(n_clusters =16)
kmeans_16.fit(data_cocaine_ready)

kmeans_17 = KMeans(n_clusters =17)
kmeans_17.fit(data_cocaine_ready)

kmeans_18 = KMeans(n_clusters =18)
kmeans_18.fit(data_cocaine_ready)

kmeans_19 = KMeans(n_clusters =19)
kmeans_19.fit(data_cocaine_ready)

kmeans_20 = KMeans(n_clusters =20)
kmeans_20.fit(data_cocaine_ready)

print("2 Cluster memberships:\n{}".format(kmeans_2.labels_))
print("3 Cluster memberships:\n{}".format(kmeans_3.labels_))
print("4 Cluster memberships:\n{}".format(kmeans_4.labels_))
print("5 Cluster memberships:\n{}".format(kmeans_5.labels_))
print("6 Cluster memberships:\n{}".format(kmeans_6.labels_))
print("7 Cluster memberships:\n{}".format(kmeans_7.labels_))
print("8 Cluster memberships:\n{}".format(kmeans_8.labels_))
print("9 Cluster memberships:\n{}".format(kmeans_9.labels_))
print("10 Cluster memberships:\n{}".format(kmeans_10.labels_))
print("11 Cluster memberships:\n{}".format(kmeans_11.labels_))
print("12 Cluster memberships:\n{}".format(kmeans_12.labels_))
print("13 Cluster memberships:\n{}".format(kmeans_13.labels_))
print("14 Cluster memberships:\n{}".format(kmeans_14.labels_))
print("15 Cluster memberships:\n{}".format(kmeans_15.labels_))
print("16 Cluster memberships:\n{}".format(kmeans_16.labels_))
print("17 Cluster memberships:\n{}".format(kmeans_17.labels_))
print("18 Cluster memberships:\n{}".format(kmeans_18.labels_))
print("19 Cluster memberships:\n{}".format(kmeans_19.labels_))
print("20 Cluster memberships:\n{}".format(kmeans_20.labels_))


# Build clustering models for control group. 
kmeans_2 = KMeans(n_clusters =2)
kmeans_2.fit(data_control_ready)

kmeans_3 = KMeans(n_clusters =3)
kmeans_3.fit(data_control_ready)

kmeans_4 = KMeans(n_clusters =4)
kmeans_4.fit(data_control_ready)

kmeans_5 = KMeans(n_clusters =5)
kmeans_5.fit(data_control_ready)

kmeans_6 = KMeans(n_clusters =6)
kmeans_6.fit(data_control_ready)

kmeans_7 = KMeans(n_clusters =7)
kmeans_7.fit(data_control_ready)

kmeans_8 = KMeans(n_clusters =8)
kmeans_8.fit(data_control_ready)

kmeans_9 = KMeans(n_clusters =9)
kmeans_9.fit(data_control_ready)

kmeans_10 = KMeans(n_clusters =10)
kmeans_10.fit(data_control_ready)

kmeans_11 = KMeans(n_clusters =11)
kmeans_11.fit(data_control_ready)

kmeans_12 = KMeans(n_clusters =12)
kmeans_12.fit(data_control_ready)

kmeans_13 = KMeans(n_clusters =13)
kmeans_13.fit(data_control_ready)

kmeans_14 = KMeans(n_clusters =14)
kmeans_14.fit(data_control_ready)

kmeans_15 = KMeans(n_clusters =15)
kmeans_15.fit(data_control_ready)

kmeans_16 = KMeans(n_clusters =16)
kmeans_16.fit(data_control_ready)

kmeans_17 = KMeans(n_clusters =17)
kmeans_17.fit(data_control_ready)

kmeans_18 = KMeans(n_clusters =18)
kmeans_18.fit(data_control_ready)

kmeans_19 = KMeans(n_clusters =19)
kmeans_19.fit(data_control_ready)

print("2 Cluster memberships:\n{}".format(kmeans_2.labels_))
print("3 Cluster memberships:\n{}".format(kmeans_3.labels_))
print("4 Cluster memberships:\n{}".format(kmeans_4.labels_))
print("5 Cluster memberships:\n{}".format(kmeans_5.labels_))
print("6 Cluster memberships:\n{}".format(kmeans_6.labels_))
print("7 Cluster memberships:\n{}".format(kmeans_7.labels_))
print("8 Cluster memberships:\n{}".format(kmeans_8.labels_))
print("9 Cluster memberships:\n{}".format(kmeans_9.labels_))
print("10 Cluster memberships:\n{}".format(kmeans_10.labels_))
print("11 Cluster memberships:\n{}".format(kmeans_11.labels_))
print("12 Cluster memberships:\n{}".format(kmeans_12.labels_))
print("13 Cluster memberships:\n{}".format(kmeans_13.labels_))
print("14 Cluster memberships:\n{}".format(kmeans_14.labels_))
print("15 Cluster memberships:\n{}".format(kmeans_15.labels_))




""""""""" 
    Identify optimal number of clusters for cocaine group from Rousseeuw, P.J. (1987). 
    Silhouettes: a Graphical Aid to the Interpretation and Validation of Cluster Analysis. 
    Computational and Applied Mathematics 20: 53–65. doi:10.1016/0377-0427(87)90125-7.
"""""""""
Sum_of_squared_distances_cocaine = []
Silhouette_Score_cocaine = []
K_cocaine = range(2, 20)
for k in K_cocaine:
    km = KMeans(n_clusters = k, random_state=1).fit(data_cocaine_ready)
    labels = km.labels_
    silhouette = metrics.silhouette_score(data_cocaine_ready, labels, metric='euclidean')
    Sum_of_squared_distances_cocaine.append(km.inertia_)
    Silhouette_Score_cocaine.append(silhouette)

# Identify optimal number of clusters for control group
Sum_of_squared_distances_control = []
Silhouette_Score_control = []
K_control = range(2, 16)
for k in K_control:
    km = KMeans(n_clusters = k, random_state=1).fit(data_control_ready)
    labels = km.labels_
    silhouette = metrics.silhouette_score(data_control_ready, labels, metric='euclidean')
    Sum_of_squared_distances_control.append(km.inertia_)
    Silhouette_Score_control.append(silhouette)

# Create elbow plot for cocaine group. 
plt.plot(K_cocaine, Sum_of_squared_distances_cocaine, 'o-', color='black')
plt.xlabel('Number of Clusters')
plt.xticks(np.arange(0, 20, step=2))
plt.yticks(np.arange(0, 9000, step=2000))
plt.ylabel('Inertia')
plt.title('Cocaine Group')

# Create plot of silhouette scores for cocaine group
plt.plot(K_cocaine, Silhouette_Score_cocaine, 'o-', color='black')
plt.xlabel('Number of Clusters')
plt.xticks(np.arange(0, 20, step=2))
plt.yticks(np.arange(0, 0.6, step=0.1))
plt.ylabel('Silhouette Score')
plt.title('Cocaine Group')

# Create elbow plot for control group. 
plt.plot(K_control, Sum_of_squared_distances_control, 'o-', color='black')
plt.xlabel('Number of Clusters')
plt.xticks(np.arange(0, 20, step=2))
plt.ylabel('Inertia')
plt.yticks(np.arange(0, 9000, step=2000))
plt.title('Control Group')

# Create plot of silhouette scores for control group
plt.plot(K_control, Silhouette_Score_control, 'o-', color='black')
plt.xlabel('Number of Clusters')
plt.xticks(np.arange(0, 20, step=2))
plt.yticks(np.arange(0, 0.6, step=0.1))
plt.ylabel('Silhouette Score')
plt.title('Control Group')

# Show cluster scores. 
list(Silhouette_Score_cocaine)
list(Silhouette_Score_control)
