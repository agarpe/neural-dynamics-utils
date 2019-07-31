import numpy as np
import matplotlib.pyplot as plt


file_name = "../model/n2v.txt"
data = np.loadtxt(file_name,skiprows=1)

plt.plot(data[:,0],data[:,1])
plt.show()