python3 superpos_from_events.py -p1 ../../data/laser/electrical/26-May-2021/events/exp1_control_RPD2_waveform.txt -p2 ../../data/laser/electrical/26-May-2021/events/exp1_control_VD1_waveform.txt -w 30 -l1 "RPD2 control" -l2 "VD1 control" -c2 r -ti "RPD2 control VD1 control"

python3 superpos_from_events.py -p1 ../../data/laser/electrical/26-May-2021/events/exp1_control_RPD2_waveform.txt -p2 ../../data/laser/electrical/26-May-2021/events/exp1_RPD2_laser_RPD2_waveform.txt -w 30 -l1 "RPD2 control" -l2 "RPD2 laser on RPD2" -c2 r -ti "RPD2 control-laser laser on RPD2"

python3 superpos_from_events.py -p1 ../../data/laser/electrical/26-May-2021/events/exp1_control_VD1_waveform.txt -p2 ../../data/laser/electrical/26-May-2021/events/exp1_RPD2_laser_VD1_waveform.txt -w 30 -l1 "VD1 control" -l2 "VD1 laser" -c2 r -ti "VD1 control-laser laser on RPD2"

python3 superpos_from_events.py -p1 ../../data/laser/electrical/26-May-2021/events/exp1_RPD2_laser_RPD2_waveform.txt -p2 ../../data/laser/electrical/26-May-2021/events/exp1_RPD2_laser_VD1_waveform.txt -w 30 -l1 "RPD2 laser on RPD2" -l2 "VD1 laser on RPD2" -c2 r -ti "RPD2 VD1 laser laser on RPD2"

python3 superpos_from_events.py -p1 ../../data/laser/electrical/26-May-2021/events/exp1_control_RPD2_waveform.txt -p2 ../../data/laser/electrical/26-May-2021/events/exp1_gap_laser_RPD2_waveform.txt -w 30 -l1 "RPD2 control" -l2 "RPD2 laser on gap" -c2 r -ti "RPD2 control-laser laser on gap"

python3 superpos_from_events.py -p1 ../../data/laser/electrical/26-May-2021/events/exp1_control_VD1_waveform.txt -p2 ../../data/laser/electrical/26-May-2021/events/exp1_gap_laser_VD1_waveform.txt -w 30 -l1 "VD1 control" -l2 "VD1 laser on gap" -c2 r -ti "VD1 control-laser laser on gap"

python3 superpos_from_events.py -p1 ../../data/laser/electrical/26-May-2021/events/exp1_gap_laser_RPD2_waveform.txt -p2 ../../data/laser/electrical/26-May-2021/events/exp1_gap_laser_VD1_waveform.txt -w 30 -l1 "RPD2 laser on gap" -l2 "VD1 laser on gap" -c2 r -ti "RPD2 VD1 laser laser on gap"

python3 superpos_from_events.py -p1 ../../data/laser/electrical/26-May-2021/events/exp1_control_RPD2_waveform.txt -p2 ../../data/laser/electrical/26-May-2021/events/exp1_VD1_laser_RPD2_waveform.txt -w 30 -l1 "RPD2 control" -l2 "RPD2 laser on VD1" -c2 r -ti "RPD2 control-laser laser on VD1"

python3 superpos_from_events.py -p1 ../../data/laser/electrical/26-May-2021/events/exp1_control_VD1_waveform.txt -p2 ../../data/laser/electrical/26-May-2021/events/exp1_VD1_laser_VD1_waveform.txt -w 30 -l1 "VD1 control" -l2 "VD1 laser on VD1" -c2 r -ti "VD1 control-laser laser on VD1"

python3 superpos_from_events.py -p1 ../../data/laser/electrical/26-May-2021/events/exp1_VD1_laser_RPD2_waveform.txt -p2 ../../data/laser/electrical/26-May-2021/events/exp1_VD1_laser_VD1_waveform.txt -w 30 -l1 "RPD2 laser on VD1" -l2 "VD1 laser on VD1" -c2 r -ti "RPD2 VD1 laser laser on VD1"



