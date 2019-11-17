import numpy as np
import matplotlib.pyplot as plt
import sys
import os


if len(sys.argv) >1:
	file_name = sys.argv[1]
else:
	file_name = "complete.asc"

print("Ploting file from ",file_name)

# path = "./data/"+file_name
path = file_name


f = open(path)
headers = f.readline().split()
print(headers)
f.close()

# data = np.loadtxt(path,skiprows=1)


import pandas as pd

data = pd.read_csv(path, delimiter = " ", names=headers,skiprows=1)


# print("Read!")
rows = data.shape[1]
# rows = len(headers)

colors = ['red', 'black', 'blue', 'brown', 'green']

plt.figure(figsize=(25,10))
for i in range(1,rows):
	# print(i)
	plt.subplot(rows,1,i)
	# plt.plot(data[1:,0],data[1:,i])
	plt.plot(data['t'],data[headers[i]],color=colors[i%len(colors)])
	plt.title(headers[i])

# fig, axes = plt.subplots(nrows=rows, ncols=1)
# fig.tight_layout()
# plt.subplots_adjust(hspace = .001)
plt.tight_layout()
# plt.savefig("./images/"+file_name[:-3]+"png")
plt.show()




# for i in range(1,rows):
# 	print(i)
# 	plt.figure(figsize=(20,10))
# 	plt.plot(data[1:,0],data[1:,i])
# 	plt.title(headers[i])

# 	plt.savefig("./images/"+headers[i]+"activity"+".png")
# 	plt.show()
	
# os.system("rm "+path)

# for i in range(1,4):
# 	plt.plot(data[1:,0],data[1:,i],label=headers[i])
# 	plt.legend()

# plt.show()



# plt.subplot(2,1,1)
# plt.plot(data[1:,0],data[1:,1])
# plt.title("Irregular")
# plt.subplot(2,1,2)
# plt.plot(data[1:,0],data[1:,3])
# plt.title("Regular")
# plt.show()
