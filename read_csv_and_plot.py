import csv
import pandas as pd
from numpy import genfromtxt
import matplotlib.pyplot as plt

newton_data = genfromtxt('./newton_data/linear_elasticity/hist.csv', delimiter=',')
nn_data_250 = genfromtxt('./nn_data/linear_elasticity/residuals_epoch_250.csv', delimiter=',')
nn_data_300 = genfromtxt('./nn_data/linear_elasticity/residuals_epoch_300.csv', delimiter=',')

newton_energy = [newton_data[4*i, 0] for i in range(1,119)]
nn_energy_300 = [nn_data_300[i, 0] for i in range(1, 119)]
nn_energy_250 = [nn_data_250[i, 0] for i in range(1, 119)]
newton_method_residual = [newton_data[4*i, 1] for i in range(1,119)]
nn_residual_300 = [nn_data_300[i, 1] for i in range(1, 119)]
nn_residual_250 = [nn_data_250[i, 1] for i in range(1, 119)]

nn_energy = []
for i in range(1,7):
  num = i*50
  data = genfromtxt('./nn_data/linear_elasticity/residuals_epoch_' + str(num)+'.csv', delimiter=',')
  nn_energy.append(data[1, 0])

x_label = [50,100, 150,200, 250, 300]
plt.plot(x_label,nn_energy)
plt.show()

plt.figure()
plt.plot(newton_energy)
plt.plot(nn_energy_250)
plt.plot(nn_energy_300)
plt.ylabel('Newton energy')

plt.legend(['Newtons Method', 'Neural Network 250','Neural Network 300'])
plt.figure()

plt.plot(newton_method_residual)
plt.plot(nn_residual_250)
plt.plot(nn_residual_300)
plt.ylabel('Newton Residual')
plt.legend(['Newtons Method', 'Neural Network 250','Neural Network 300'])
plt.show()

