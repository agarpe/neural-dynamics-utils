import numpy as np 
import sys
import matplotlib.pyplot as plt


if len(sys.argv) > 2:
	path = sys.argv[1]
	file = sys.argv[2]
	filename = path + file
else:
	path = '../../../TFM/WangModel/data/selection2/'
	filename  = 'global_stats.txt'

file = path+filename

# tau_fac A_syn tau_rec tau_in
# 200 75 50 2
# Type Acc Dec
# Spikes_after_1nd 11 2
# Spikes_after_2nd 5 19
# Spikes_after_3nd 26 23
# end

f = open(file)
line = f.readline() 

index =1

x = []
ticks = []
while(line):
	param_headers = line.split()
	param_values = ' '.join(str(f.readline()).split())
	print(param_values)
	result_headers = f.readline().split()

	l = f.readline()
	print(l)
	if(l != 'end'):
		spk1 = f.readline().split()[1:]
		l = f.readline()	
		if(l != 'end'):
			spk2 = f.readline().split()[1:]
			l = f.readline()
			if(l != 'end'):
				spk3 = f.readline().split()[1:]
			else:
				spk3=[]
		else:
			spk2=[]
		
	else:
		print(l)
		spk1=[]


	end = f.readline()
	print(param_values,"\n")
	#print(spk1)
	#print(index)
	try:
		plt.bar(index,spk1[1],width=0.1,color='g')
		ticks.append(param_values)
		x+=1
	except:
		pass
	try:
		plt.bar(index+0.1,spk2[1],width=0.1,color='r')
		ticks.append(param_values)
		x+=1
	except:
		pass
	try:
		plt.bar(index+0.2,spk3[1],width=0.1,color='g')
		ticks.append(param_values)
		x+=1
	except:
		pass


	line = f.readline() 
	index += 1


print(ticks)
plt.xticks(range(1,x),ticks)
plt.show()


f.close()

