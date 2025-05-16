import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
# RPD2 stimulation
# intra0 intra1 curr2 curr3

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="Path to the file to analyze")
ap.add_argument("-s", "--scale", required=True, help="Data in column c will be divided by this value")
ap.add_argument("-c", "--column", required=False,default=0, help="Column")

args = vars(ap.parse_args())

col = int(args['column'])
path = args['path']
scale = float(args['scale'])

df = pd.read_csv(path, delimiter = "	",skiprows=1,header=None)


print(df.head())
# data=df.drop([4], axis=1)
# data=data.drop([0])
data=df.values[:]
# data = df
print(type(data))
# print(data.head())
print(data.shape)
# data = data[:-1]


plt.plot(data[col])

data[col]=data[col]/scale
print(data.shape)

open(path,'w')
np.savetxt(path,data,delimiter=' ')

plt.plot(data[col])
plt.show()
