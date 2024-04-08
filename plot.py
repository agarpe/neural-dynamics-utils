# Developed by Alicia Garrido Peña (2023)
#
# Plotting script for signals
#
# 
############################################################################################

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({'font.size': 50})

headers = []
reduced_headers = []
if len(sys.argv) >3:
	reduced_headers = sys.argv[3].split()
if len(sys.argv) >2:
	path = sys.argv[1]
	file_name = sys.argv[2]
else:
	print("Error: No file specified \n Format: <path> <file_name>")
	exit()

print("Ploting file from ",file_name)

path = path+file_name

if headers == []:
	f = open(path)
	f.readline()
	headers = f.readline().split()
	print(headers)
	try:
		headers.remove('#')
	except:
		pass
	f.close()

print(headers)
data = pd.read_csv(path, delimiter = " ", dtype=np.float64,names=headers,skiprows=2,low_memory=False)

if len(reduced_headers) >0:
	rows = len(reduced_headers)
else:
	rows = data.shape[1]
	reduced_headers = headers

# colors = ['teal', 'lightsalmon', 'skyblue', 'darkseagreen','maroon','teal', 'brown', 'blue', 'green','maroon']
colors = ['teal', 'lightsalmon', 'darkseagreen','maroon','teal', 'brown', 'blue', 'green','maroon']
colors = ['teal','cornflowerblue', 'darkblue', 'teal','teal']

colors = plt.cm.tab20(np.linspace(0,1,len(headers)))

ini = 1
if 't' not in headers:
	data['t'] = np.arange(data[headers[0]].size)*0.1
	ini = 0


# data[reduced_headers].plot()
# plt.show()


data = data[['t']+reduced_headers].values.T
fig,axes = plt.subplots(nrows=len(reduced_headers),sharex=True,figsize=(30,30))
for i,d in enumerate(data[1:]):
	# plt.plot(data['t'],d)
	axes[i].plot(data[0], d, color = colors[i])


# plt.xlim(37000,38700) # 22-02
# plt.xlim(61600,62700) #depol 15h30m59s_Trial7_10-05-2022_depol.asc
# plt.show()

# plt.savefig("./images/"+file_name[:-4]+'.pdf',format='pdf')
plt.show()
exit()

# rows = 
plt.figure(figsize=(30,30))
for i in range(ini,rows):
	if(i==ini):
		ax1 = plt.subplot(rows-1,1,i+1-ini)
	else:	
		plt.subplot(rows-1,1,i,sharex=ax1)

	if(i==rows-1):
		plt.xlabel("Time (ms)")

	if(headers[i]=='c'):
		plt.ylabel("Current", multialignment='center')	
	else:
		print(reduced_headers[i])
		print(data[reduced_headers[i]])
		plt.ylabel("Voltage\n(mV)", multialignment='center')
	plt.plot(data['t'],data[reduced_headers[i]],color=colors[i-1])

	plt.title(reduced_headers[i])

plt.tight_layout()
plt.show()
# plt.xlim(0,10000)
# plt.savefig("./images/"+file_name[:-3]+"eps",format='eps')
# plt.show()

# print(data)
# print(data.shape)

# data[60000:][:100000]

plt.figure(figsize=(30,10))
for i in range(ini,rows):
	# if(i==1):
	# 	ax1 = plt.subplot(rows,1,i)
	# else:	
	# 	plt.subplot(rows,1,i,sharex=ax1)

	if reduced_headers[i] == 'SO' or reduced_headers[i] == 'c':
		continue
	if(i==rows-1):
		plt.xlabel("Time (ms)")

	if(reduced_headers[i]!='c' and reduced_headers[i]!='SO'):
		plt.ylabel("Voltage\n(mV)", multialignment='center')
		try:
			plt.plot(data['t'],data[reduced_headers[i]],color=colors[i-1],label=reduced_headers[i],linewidth=2)
		except:
			plt.plot(data[reduced_headers[i]],color=colors[i-1],label=reduced_headers[i],linewidth=2)

	plt.title(reduced_headers[i])

# plt.xlim(6000,10000)

# plt.legend()
plt.tight_layout()

# plt.savefig("./images/"+file_name[:-3]+"eps",format='eps')
plt.show()

