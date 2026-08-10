# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 17:09:45 2025

@author: Herman
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,1,10001)
y = np.exp(-x/8)

noise = np.random.normal(-0.000, 0.000, 10001)


tau = np.logspace(0,1,500)
print(tau)


def my_function(x, tau):
    return np.exp(-x/tau)



# 3. Create a grid of x and y coordinates
X, Y = np.meshgrid(tau, x)

Z = []
for X in x:
    row = []
    for Tau in tau:
        row.append(np.exp(-X/Tau))
    Z.append(row)


# 4. Apply the function element-wise
# Z = my_function(X, Y)
Z = np.array(Z)

# print(len(Z[0,:]))

# Print the resulting matrix (optional)
# print(Z)


# r, residuals, rank, s = np.linalg.lstsq(Z, y, rcond=None)
# r = np.array(r)
# print(len(r))
# print(len(Z[0,:]))
# # r_t = [0,0,0,0,0,1,0,0,0,0,0]
# plt.plot(x,Z.dot(r))
# # plt.plot(x,Z.dot(r_t))
# plt.plot(x,y)
# plt.show()

# plt.plot(tau,r)


from sklearn.linear_model import Ridge


# Create Ridge Regression model
ridge = Ridge(alpha=1)  # alpha is the regularization strength

# Fit the model
ridge.fit(Z, y+noise)

# Predictions
# y_pred = ridge.predict(Z)

# Evaluate performance
# mse = mean_squared_error(y_test, y_pred)

print("Coefficients:", ridge.coef_)
plt.plot(tau,ridge.coef_)
# print("Intercept:", ridge.intercept_)
# print("Mean Squared Error:", mse)


# print(Z.dot(r_t))
# print(Z[:,5])
# plt.plot(x,Z[:,5])
# for i in range(0,4):
#     print(Z[i,0]*r[0]+Z[i,1]*r[1])
# # print(y)
# plt.plot(tau,m[0])
# plt.xscale('log')
# # 