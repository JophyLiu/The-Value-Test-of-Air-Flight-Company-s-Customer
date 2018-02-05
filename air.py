# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 00:23:50 2018

@author: LIU
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

air_data=pd.read_csv('air_data.csv',encoding='utf-8')

shape=air_data.shape
column=air_data.columns
print(column)

explore=air_data.describe(percentiles=[],include='all').T
explore['null']=len(air_data)-explore['count']
explore=explore[['null','max','min']]
print(explore)

null_data=air_data[air_data.isnull()]
air_data=air_data[air_data['SUM_YR_1'].notnull()]
air_data=air_data[air_data['SUM_YR_1'].notnull()]

index1=air_data['SUM_YR_1'] != 0
index2=air_data['SUM_YR_2'] != 0
index3=(air_data['SEG_KM_SUM']==0) & (air_data['avg_discount']==0)
air_data=air_data[index1 | index2 | index3]

new_data=air_data[['FFP_DATE','LOAD_TIME','FLIGHT_COUNT','avg_discount','SEG_KM_SUM','LAST_TO_END']]
new_data['AVG_DISCOUNT']=new_data['avg_discount']
new_data=new_data[['FFP_DATE','LOAD_TIME','FLIGHT_COUNT','AVG_DISCOUNT','SEG_KM_SUM','LAST_TO_END']]


new_data['L']=pd.to_datetime(new_data['LOAD_TIME']) - pd.to_datetime(new_data['FFP_DATE'])
new_data['R']=new_data['LAST_TO_END']
new_data['F']=new_data['FLIGHT_COUNT']
new_data['M']=new_data['SEG_KE_SUM']
new_data['C']=new_data['AVG_DISCOUNT']

new_data=new_data[['L','R','F','M','C']]

standard_data=(new_data-new_data.mean(axis = 0)) /(new_data.std(axis = 0))
standard_data.columns=['Z'+i for i in standard_data.columns]


from sklearn.cluster import KMeans
standard_data=pd.read_excel('zscoreddata.xls')
x=standard_data.iloc[:,:].values


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', n_jobs=4, random_state = 42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters = 5, init = 'k-means++', n_jobs=4, random_state = 42)
y_kmeans = kmeans.fit_predict(x)


kmeans.cluster_centers_ 
kmeans.labels_


labels = standard_data.columns 
k = 5 
plot_data = kmeans.cluster_centers_
color = ['b', 'g', 'r', 'c', 'y'] 

angles = np.linspace(0, 2*np.pi, k, endpoint=False)
plot_data = np.concatenate((plot_data, plot_data[:,[0]]), axis=1) 
angles = np.concatenate((angles, [angles[0]])) 

fig = plt.figure()
ax = fig.add_subplot(111, polar=True) 
for i in range(len(plot_data)):
  ax.plot(angles, plot_data[i], 'o-', color = color[i], label = 'costomer'+str(i), linewidth=2)

ax.set_rgrids(np.arange(0.01, 3.5, 0.5), np.arange(-1, 2.5, 0.5), fontproperties="SimHei")
ax.set_thetagrids(angles * 180/np.pi, labels, fontproperties="SimHei")
plt.legend(loc = 4)
plt.show()






