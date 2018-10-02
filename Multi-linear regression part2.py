import numpy as np
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from math import exp, pow
import pandas as pd
from math import pi
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
print(sample_100.head())

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
print(sample_1000.head())

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
print(sample_10000.head())

training_10000=sample_10000.sample(frac=0.7)
testing_10000=sample_10000.sample(frac=0.3)

#==============================================================

xxx, yyy = np.meshgrid(np.linspace(0,10,10), np.linspace(0,10,10))
XX = np.column_stack((xxx.flatten(),yyy.flatten()))







X=training_100.iloc[:,0:2]
X=X.as_matrix()
Z=training_100.iloc[:,2]
Z=Z.as_matrix()

# 建立线性回归模型
regr = linear_model.LinearRegression() 
# 拟合
regr.fit(X, Z) 
xx = training_100.iloc[:,0]
xx=xx.as_matrix()
yy = training_100.iloc[:,1]
yy=yy.as_matrix()
zz = training_100.iloc[:,2]
zz=zz.as_matrix()

fig = plt.figure()
ax = fig.gca(projection='3d')
# 1.画出真实的点
ax.scatter(xx, yy, zz)
ax.plot_wireframe(xxx, yyy, regr.predict(XX).reshape(10,10))
ax.plot_surface(xxx, yyy, regr.predict(XX).reshape(10,10), alpha=0.3)
plt.show()
#=========================================================
X=training_1000.iloc[:,0:2]
X=X.as_matrix()
Z=training_1000.iloc[:,2]
Z=Z.as_matrix()

# 建立线性回归模型
regr = linear_model.LinearRegression() 
# 拟合
regr.fit(X, Z) 
# 不难得到平面的系数、截距
a, b = regr.coef_, regr.intercept_ 
#画图
xx = training_1000.iloc[:,0]
xx=xx.as_matrix()
yy = training_1000.iloc[:,1]
yy=yy.as_matrix()
zz = training_1000.iloc[:,2]
zz=zz.as_matrix()

fig = plt.figure()
ax = fig.gca(projection='3d')
# 1.画出真实的点
ax.scatter(xx, yy, zz)
ax.plot_wireframe(xxx, yyy, regr.predict(XX).reshape(10,10))
ax.plot_surface(xxx, yyy, regr.predict(XX).reshape(10,10), alpha=0.3)
plt.show()

#=========================================================

X=training_10000.iloc[:,0:2]
X=X.as_matrix()
Z=training_10000.iloc[:,2]
Z=Z.as_matrix()
"""
print(type(X))
print(type(Z))
print(X.shape)
print(Z.shape)
"""
# 建立线性回归模型
regr = linear_model.LinearRegression() 
# 拟合
regr.fit(X, Z) 
#画图
xx = training_10000.iloc[:,0]
xx=xx.as_matrix()
yy = training_10000.iloc[:,1]
yy=yy.as_matrix()
zz = training_10000.iloc[:,2]
zz=zz.as_matrix()

fig = plt.figure()
ax = fig.gca(projection='3d')
# 1.画出真实的点
ax.scatter(xx, yy, zz)
ax.plot_wireframe(xxx, yyy, regr.predict(XX).reshape(10,10))
ax.plot_surface(xxx, yyy, regr.predict(XX).reshape(10,10), alpha=0.3)
plt.show()








