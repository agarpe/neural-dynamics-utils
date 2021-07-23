
python3 superpos_from_events.py -p1 ../../data/laser/electrical/26-May-2021/events/$1/Exp1_Control_RPD2_Waveform.txt -p2 ../../data/laser/electrical/26-May-2021/events/$1/Exp1_Control_VD1_Waveform.txt -w 30 -l1 "RPD2 Control" -l2 "VD1 Control" -c2 r -ti "RPD2 Control VD1 Control" -sh n

python3 superpos_from_events.py -p1 ../../data/laser/electrical/26-May-2021/events/$1/Exp1_Control_RPD2_Waveform.txt -p2 ../../data/laser/electrical/26-May-2021/events/$1/Exp1_RPD2_Laser_RPD2_Waveform.txt -w 30 -l1 "RPD2 Control" -l2 "RPD2 Laser on RPD2" -c2 r -ti "RPD2 Control-Laser Laser on RPD2" -sh n

python3 superpos_from_events.py -p1 ../../data/laser/electrical/26-May-2021/events/$1/Exp1_Control_VD1_Waveform.txt -p2 ../../data/laser/electrical/26-May-2021/events/$1/Exp1_RPD2_Laser_VD1_Waveform.txt -w 30 -l1 "VD1 Control" -l2 "VD1 Laser" -c2 r -ti "VD1 Control-Laser Laser on RPD2" -sh n

python3 superpos_from_events.py -p1 ../../data/laser/electrical/26-May-2021/events/$1/Exp1_RPD2_Laser_RPD2_Waveform.txt -p2 ../../data/laser/electrical/26-May-2021/events/$1/Exp1_RPD2_Laser_VD1_Waveform.txt -w 30 -l1 "RPD2 Laser on RPD2" -l2 "VD1 Laser on RPD2" -c2 r -ti "RPD2 VD1 Laser Laser on RPD2" -sh n

python3 superpos_from_events.py -p1 ../../data/laser/electrical/26-May-2021/events/$1/Exp1_Control_RPD2_Waveform.txt -p2 ../../data/laser/electrical/26-May-2021/events/$1/Exp1_Gap_Laser_RPD2_Waveform.txt -w 30 -l1 "RPD2 Control" -l2 "RPD2 Laser on Gap" -c2 r -ti "RPD2 Control-Laser Laser on Gap" -sh n

python3 superpos_from_events.py -p1 ../../data/laser/electrical/26-May-2021/events/$1/Exp1_Control_VD1_Waveform.txt -p2 ../../data/laser/electrical/26-May-2021/events/$1/Exp1_Gap_Laser_VD1_Waveform.txt -w 30 -l1 "VD1 Control" -l2 "VD1 Laser on Gap" -c2 r -ti "VD1 Control-Laser Laser on Gap" -sh n

python3 superpos_from_events.py -p1 ../../data/laser/electrical/26-May-2021/events/$1/Exp1_Gap_Laser_RPD2_Waveform.txt -p2 ../../data/laser/electrical/26-May-2021/events/$1/Exp1_Gap_Laser_VD1_Waveform.txt -w 30 -l1 "RPD2 Laser on Gap" -l2 "VD1 Laser on Gap" -c2 r -ti "RPD2 VD1 Laser Laser on Gap" -sh n

python3 superpos_from_events.py -p1 ../../data/laser/electrical/26-May-2021/events/$1/Exp1_Control_RPD2_Waveform.txt -p2 ../../data/laser/electrical/26-May-2021/events/$1/Exp1_VD1_Laser_RPD2_Waveform.txt -w 30 -l1 "RPD2 Control" -l2 "RPD2 Laser on VD1" -c2 r -ti "RPD2 Control-Laser Laser on VD1" -sh n

python3 superpos_from_events.py -p1 ../../data/laser/electrical/26-May-2021/events/$1/Exp1_Control_VD1_Waveform.txt -p2 ../../data/laser/electrical/26-May-2021/events/$1/Exp1_VD1_Laser_VD1_Waveform.txt -w 30 -l1 "VD1 Control" -l2 "VD1 Laser on VD1" -c2 r -ti "VD1 Control-Laser Laser on VD1" -sh n

python3 superpos_from_events.py -p1 ../../data/laser/electrical/26-May-2021/events/$1/Exp1_VD1_Laser_RPD2_Waveform.txt -p2 ../../data/laser/electrical/26-May-2021/events/$1/Exp1_VD1_Laser_VD1_Waveform.txt -w 30 -l1 "RPD2 Laser on VD1" -l2 "VD1 Laser on VD1" -c2 r -ti "RPD2 VD1 Laser Laser on VD1" -sh n



