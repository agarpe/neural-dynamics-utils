#!/bin/bash
python3 burst_charact.py ../articulo/data/ definitivo_n3t_Euler_0.0100_9.000_10.000_1.000_-1.000_4.600_0.250_0.000_5.000 N3t $1;
python3 burst_charact.py ../articulo/data/ definitivo_n3t_Euler_0.0010_9.000_10.000_1.000_-1.000_4.600_0.250_0.000_5.000 N3t $1;
python3 burst_charact.py ../articulo/data/ definitivo_n3t_Runge-Kutta_0.0050_9.000_10.000_1.000_-1.000_4.600_0.250_0.000_5.000 N3t $1;
python3 burst_charact.py ../articulo/data/ definitivo_n3t_Runge-Kutta_0.0080_9.000_10.000_1.000_-1.000_4.600_0.250_0.000_5.000 N3t $1;