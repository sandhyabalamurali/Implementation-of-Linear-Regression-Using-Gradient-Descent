# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

     1.Import the required library and read the dataframe.

     2.Write a function computeCost to generate the cost function.

     3.Perform iterations og gradient steps with learning rate.

     4.Plot the Cost function using Gradient Descent and generate the required graph.


## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: SANDHYA B N
RegisterNumber: 212222040144
# Import required package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("ex1.txt",header=None)
data
data.shape
plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Vs Prediction")
def computeCost(X,y,theta):
    m=len(y)
    h=X.dot(theta)
    square_err=(h-y)**2

    return 1/(2*m) * np.sum(square_err)
data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(X,y,theta)
theta.shape
y.shape
X.shape
def gradientDescent(X,y,theta,alpha,num_iters):
  
  m=len(y)
  J_history=[]

  for i in range(num_iters):
    predictions=X.dot(theta)
    error=np.dot(X.transpose(),(predictions - y))
    descent=alpha * 1/m * error
    theta-=descent
    J_history.append(computeCost(X,y,theta))

  return theta, J_history
  
theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")
plt.plot(J_history)
plt.xlabel("Iternations")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")
plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Polpulation of City (10,000s)")
plt.ylabel("Profit (10,000s)")
plt.title("Profit Prediction")
def predict(x,theta):
  predictions= np.dot(theta.transpose(),x)
  return predictions[0]
  
predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0))) 
*/
```

## Output:
![linear regression using gradient descent](sam.png)

![ml ex 3 1](https://github.com/sandhyabalamurali/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/115525118/8bd74f45-23b8-418f-936e-14d64c0e749c)



![ml ex 3 2](https://github.com/sandhyabalamurali/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/115525118/9735e987-72eb-43e7-bbf5-85a24bd79240)



![ex 3 3](https://github.com/sandhyabalamurali/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/115525118/739183ab-8d90-4abe-ad89-ece66c657800)



![4](https://github.com/sandhyabalamurali/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/115525118/396c29a5-a52a-4e70-a9e0-f2fed7d82266)



![5](https://github.com/sandhyabalamurali/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/115525118/bb8cd81a-9e27-4d48-9b0a-7575aa3f75ea)



![6](https://github.com/sandhyabalamurali/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/115525118/33f5899c-eb58-48ab-ac30-d9a9ad7f327d)



![7](https://github.com/sandhyabalamurali/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/115525118/93e69411-6a11-4389-96f0-24a3bf5cc729)



![8](https://github.com/sandhyabalamurali/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/115525118/ffbbb691-2b2f-409e-a570-35858ff42a14)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
