from sklearn import svm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from math import pow
import pandas as pd
#=============================================================
X=training_100.iloc[:,0:2]
y=training_100.iloc[:,2]
X=np.array(X)
X=X.tolist()
y=np.array(y)
y=y.tolist()
"""
print(X)
print(y)
print(len(X))
print(len(y))
print(type(X))
print(type(y))
"""
X_test=testing_100.iloc[:,0:2] #30 elements
y_test=testing_100.iloc[:,2]
X_test=np.array(X_test)
X_test=X_test.tolist()
y_test=np.array(y_test)
y_test=y_test.tolist()
"""
print(X_test)
print(y_test)
print(len(X_test))
print(len(y_test))
print(type(X_test))
print(type(y_test))
"""
clf = svm.SVR()
clf.fit(X, y)
predictions=clf.predict(X_test)

test_error=0
for i in range(len(predictions)):
    print("predict: %s, target: %s" %(predictions[i],y_test[i]))
    test_error=test_error+pow((predictions[i]-y_test[i]),2)
    
print("test error %.7f" %test_error)

#=====================

xxx, yyy = np.meshgrid(np.linspace(0,10,10), np.linspace(0,10,10))
XX = np.column_stack((xxx.flatten(),yyy.flatten()))

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
ax.plot_wireframe(xxx, yyy, clf.predict(XX).reshape(10,10))
ax.plot_surface(xxx, yyy, clf.predict(XX).reshape(10,10), alpha=0.3)
plt.show()


#=============================================================
X=training_1000.iloc[:,0:2]
y=training_1000.iloc[:,2]
X=np.array(X)
X=X.tolist()
y=np.array(y)
y=y.tolist()

X_test=testing_1000.iloc[:,0:2] #30 elements
y_test=testing_1000.iloc[:,2]
X_test=np.array(X_test)
X_test=X_test.tolist()
y_test=np.array(y_test)
y_test=y_test.tolist()

clf = svm.SVR()
clf.fit(X, y)
predictions=clf.predict(X_test)

test_error=0
for i in range(len(predictions)):
    print("predict: %s, target: %s" %(predictions[i],y_test[i]))
    test_error=test_error+pow((predictions[i]-y_test[i]),2)
    
print("test error %.7f" %test_error)
#=====================

xxx, yyy = np.meshgrid(np.linspace(0,10,10), np.linspace(0,10,10))
XX = np.column_stack((xxx.flatten(),yyy.flatten()))

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
ax.plot_wireframe(xxx, yyy, clf.predict(XX).reshape(10,10))
ax.plot_surface(xxx, yyy, clf.predict(XX).reshape(10,10), alpha=0.3)
plt.show()


#=============================================================
X=training_10000.iloc[:,0:2]
y=training_10000.iloc[:,2]
X=np.array(X)
X=X.tolist()
y=np.array(y)
y=y.tolist()

X_test=testing_10000.iloc[:,0:2] #30 elements
y_test=testing_10000.iloc[:,2]
X_test=np.array(X_test)
X_test=X_test.tolist()
y_test=np.array(y_test)
y_test=y_test.tolist()

clf = svm.SVR()
clf.fit(X, y)
predictions=clf.predict(X_test)

test_error=0
for i in range(len(predictions)):
    print("predict: %s, target: %s" %(predictions[i],y_test[i]))
    test_error=test_error+pow((predictions[i]-y_test[i]),2)
    
print("test error %.7f" %test_error)
#=====================

xxx, yyy = np.meshgrid(np.linspace(0,10,10), np.linspace(0,10,10))
XX = np.column_stack((xxx.flatten(),yyy.flatten()))

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
ax.plot_wireframe(xxx, yyy, clf.predict(XX).reshape(10,10))
ax.plot_surface(xxx, yyy, clf.predict(XX).reshape(10,10), alpha=0.3)
plt.show()
