import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from math import exp, pow
import sys
import copy
from scipy.linalg import norm, pinv


#reading data locally
df=pd.read_excel('/Users/MokiWind/Desktop/DATASET.xlsx')
print(df.head()) #show the dataframe intuitively
print(df.tail())
#type(df)  ===>the type is DataFrame
#print(df.ix[0:2,0:3])

df=df.drop(df.columns[[0,3]], axis = 1)

print (df.describe()) #carry out some basic descriptive statistcal analysis
#==============================================================
#get three set of data of three different volume level 
sample_100=df.sample(frac=1).iloc[0:200,:]
sample_100.iloc[:,2]=sample_100.iloc[:,2]/100000
sample_100.iloc[:,1]=sample_100.iloc[:,1]*1.0
sample_100.iloc[:,0]=sample_100.iloc[:,0]*1.0
sample_100=sample_100.sort_values(by=['Price'])
sample_100=sample_100.iloc[0:190,:]
sample_100=sample_100.sample(frac=1).iloc[0:100,:]
print(sample_100)
print(sample_100.describe())
print(sample_100.corr())
print(sample_100.cov())
plt.show(sample_100.plot(kind = 'box'))
plt.show(sns.distplot(sample_100.iloc[:,0], rug = True, bins = 15))
plt.show(sns.distplot(sample_100.iloc[:,1], rug = True, bins = 1))
plt.show(sns.distplot(sample_100.iloc[:,2], rug = True, bins = 15))
plt.show(sns.lmplot("Region", "Price", sample_100))
plt.show(sns.lmplot("Category", "Price", sample_100))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X_100 = sample_100.iloc[:,0]
Y_100 = sample_100.iloc[:,1]
Z_100 = sample_100.iloc[:,2]
ax.scatter(X_100, Y_100, Z_100)
plt.show()

training_100=sample_100.sample(frac=0.7)
testing_100=sample_100.sample(frac=0.3)


#============================================================== 
sample_1000=df.sample(frac=1).iloc[0:2000,:]
sample_1000.iloc[:,2]=sample_1000.iloc[:,2]/100000
sample_1000.iloc[:,1]=sample_1000.iloc[:,1]*1.0
sample_1000.iloc[:,0]=sample_1000.iloc[:,0]*1.0
sample_1000=sample_1000.sort_values(by=['Price'])
sample_1000=sample_1000.iloc[0:1900,:]
sample_1000=sample_1000.sample(frac=1).iloc[0:1000,:]
print(sample_1000)
print(sample_1000.describe())
print(sample_1000.corr())
print(sample_1000.cov())
plt.show(sample_1000.plot(kind = 'box'))
plt.show(sns.distplot(sample_1000.iloc[:,0], rug = True, bins = 15))
plt.show(sns.distplot(sample_1000.iloc[:,1], rug = True, bins = 1))
plt.show(sns.distplot(sample_1000.iloc[:,2], rug = True, bins = 15))
plt.show(sns.lmplot("Region", "Price", sample_1000))
plt.show(sns.lmplot("Category", "Price", sample_1000))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X_1000 = sample_1000.iloc[:,0]
Y_1000 = sample_1000.iloc[:,1]
Z_1000 = sample_1000.iloc[:,2]
ax.scatter(X_1000, Y_1000, Z_1000)
plt.show()

training_1000=sample_1000.sample(frac=0.7)
testing_1000=sample_1000.sample(frac=0.3)

#===============================================================

sample_10000=df.sample(frac=1).iloc[0:20000,:]
sample_10000.iloc[:,2]=sample_10000.iloc[:,2]/100000
sample_10000.iloc[:,1]=sample_10000.iloc[:,1]*1.0
sample_10000.iloc[:,0]=sample_10000.iloc[:,0]*1.0
sample_10000=sample_10000.sort_values(by=['Price'])
sample_10000=sample_10000.iloc[0:19000,:]
sample_10000=sample_10000.sample(frac=1).iloc[0:10000,:]
print(sample_10000)
print(sample_10000.describe())
print(sample_10000.corr())
print(sample_10000.cov())
plt.show(sample_10000.plot(kind = 'box'))
plt.show(sns.distplot(sample_10000.iloc[:,0], rug = True, bins = 15))
plt.show(sns.distplot(sample_10000.iloc[:,1], rug = True, bins = 1))
plt.show(sns.distplot(sample_10000.iloc[:,2], rug = True, bins = 15))
plt.show(sns.lmplot("Region", "Price", sample_10000))
plt.show(sns.lmplot("Category", "Price", sample_10000))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X_10000 = sample_10000.iloc[:,0]
Y_10000 = sample_10000.iloc[:,1]
Z_10000 = sample_10000.iloc[:,2]
ax.scatter(X_10000, Y_10000, Z_10000)
plt.show()

training_10000=sample_10000.sample(frac=0.7)
testing_10000=sample_10000.sample(frac=0.3)

#=============================================