
from charact_utils import *

if len(sys.argv) >2:
	path = sys.argv[1]
	file_name = sys.argv[2]
else:
	print("Format: concatenate_events.py path file_name")



path+= file_name + "/"


N1m_data = read_model_burst_path(path+"N1M",scale=1)
N2v_data = read_model_burst_path(path+"N2v",scale=1)
N3t_data = read_model_burst_path(path+"N3t",scale=1)

N1m_data=trunc(N1m_data,decs=2)
N2v_data=trunc(N2v_data,decs=2)
N3t_data=trunc(N3t_data,decs=2)


N1_N2,N2_N1 = get_intervals(N1m_data,N2v_data)
period= get_burst_period(N1m_data)

file = open(path+"/events.asc","w")

#Format:
# N1M_fst N1M_lst N2v_fst N2v_lst  N3t_fst N3t_lst period N2N1delay N2N1interval
for i in range(N1m_data.shape[0]-1):
	file.write("%d %f %f %f %f %f %f %f %f %f\n"%(i,N1m_data[i,0],N1m_data[i,1],N2v_data[i,0],N2v_data[i,1],N3t_data[i,0],N3t_data[i,1],period[i],N2_N1[DELAY][i],N2_N1[INTERVAL][i]))

