from numpy.linalg import inv
from numpy import dot, transpose
import numpy as np
import pandas as pd
from math import exp, pow

"""
X=[[1,6,2],[1,8,1],[1,10,0],[1,14,2],[1,18,0]]
y=[[7],[9],[13],[17.5],[18]]
print(dot(inv(dot(transpose(X),X)),dot(transpose(X),y)))
"""
from sklearn.linear_model import LinearRegression

X=training_100.iloc[:,0:2]
y=training_100.iloc[:,2]
X=np.array(X)
X=X.tolist()
y=np.array(y)
y=y.tolist()
model=LinearRegression()
model.fit(X,y)
X_test=testing_100.iloc[:,0:2] #30 elements
y_test=testing_100.iloc[:,2]

X_test=np.array(X_test)
X_test=X_test.tolist()

y_test=np.array(y_test)
y_test=y_test.tolist()

predictions=model.predict(X_test)

test_error=0
for i in range(len(predictions)):
    print("predict: %s,target: %s" %(predictions[i],y_test[i]))
    test_error=test_error+pow((predictions[i]-y_test[i]),2)
    
print("test error %.7f" %test_error)
#===========================================================
X=training_1000.iloc[:,0:2]
y=training_1000.iloc[:,2]
X=np.array(X)
X=X.tolist()
y=np.array(y)
y=y.tolist()
model=LinearRegression()
model.fit(X,y)
X_test=testing_1000.iloc[:,0:2] #30 elements
y_test=testing_1000.iloc[:,2]

X_test=np.array(X_test)
X_test=X_test.tolist()

y_test=np.array(y_test)
y_test=y_test.tolist()

predictions=model.predict(X_test)

test_error=0
for i in range(len(predictions)):
    print("predict: %s,target: %s" %(predictions[i],y_test[i]))
    test_error=test_error+pow((predictions[i]-y_test[i]),2)
    
print("test error %.7f" %test_error)
#===========================================================
X=training_10000.iloc[:,0:2]
y=training_10000.iloc[:,2]
X=np.array(X)
X=X.tolist()
y=np.array(y)
y=y.tolist()
model=LinearRegression()
model.fit(X,y)
X_test=testing_10000.iloc[:,0:2] #30 elements
y_test=testing_10000.iloc[:,2]

X_test=np.array(X_test)
X_test=X_test.tolist()

y_test=np.array(y_test)
y_test=y_test.tolist()

predictions=model.predict(X_test)

test_error=0
for i in range(len(predictions)):
    print("predict: %s,target: %s" %(predictions[i],y_test[i]))
    test_error=test_error+pow((predictions[i]-y_test[i]),2)
    
print("test error %.7f" %test_error)
    