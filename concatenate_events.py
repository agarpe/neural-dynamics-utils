
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


N1m_data,N2v_data,N3t_data = fix_length(N1m_data,N2v_data,N3t_data)


N1_N2,N2_N1 = get_intervals(N1m_data,N2v_data)
N1_N3,N3_N1 = get_intervals(N1m_data,N3t_data)
N2_N3,N3_N2 = get_intervals(N2v_data,N3t_data)
period= get_burst_period(N1m_data)

N1_DUR = get_burst_duration(N1m_data)
N2_DUR = get_burst_duration(N2v_data)
N3_DUR = get_burst_duration(N3t_data)

file = open(path+"/events.asc","w")

#Format:
# N1M_fst N1M_lst N2v_fst N2v_lst  N3t_fst N3t_lst period N2N1delay N2N1interval
file.write("N1M_fst N1M_lst N2v_fst N2v_lst  N3t_fst N3t_lst period N1burst N2burst N3burst N1_N2_INTERVAL N1_N2_DELAY N2_N1_INTERVAL N2_N1_DELAY N1_N3_INTERVAL N1_N3_DELAY N3_N1_INTERVAL N3_N1_DELAY N2_N3_INTERVAL N2_N3_DELAY N3_N2_INTERVAL N3_N2_DELAY ")
for i in range(N1m_data.shape[0]-1):
	file.write("%d %f %f %f %f %f %f "%(i,N1m_data[i,0],N1m_data[i,1],N2v_data[i,0],N2v_data[i,1],N3t_data[i,0],N3t_data[i,1]))
	file.write("%f %f %f %f "%(period[i],N1_DUR[i],N2_DUR[i],N3_DUR[i]))
	file.write("%f %f %f %f "%(N1_N2[INTERVAL][i],N1_N2[DELAY][i],N2_N1[INTERVAL][i],N2_N1[DELAY][i]))
	file.write("%f %f %f %f "%(N1_N3[INTERVAL][i],N1_N3[DELAY][i],N3_N1[INTERVAL][i],N3_N1[DELAY][i]))
	file.write("%f %f %f %f\n"%(N2_N3[INTERVAL][i],N2_N3[DELAY][i],N3_N2[INTERVAL][i],N3_N2[DELAY][i]))

