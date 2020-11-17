#!/bin/bash
python3 burst_charact.py ../articulo/data/ definitivo_so_Euler_0.0100_-1.000_10.000_1.000_4.000_4.600_0.250_8.200_13.000 SO $1;
python3 burst_charact.py ../articulo/data/ definitivo_so_Euler_0.0010_-1.000_10.000_1.000_4.000_4.600_0.250_8.200_13.000 SO $1;
python3 burst_charact.py ../articulo/data/ definitivo_so_Runge-Kutta_0.0050_-1.000_10.000_1.000_4.000_4.600_0.250_8.200_13.000 SO $1;
python3 burst_charact.py ../articulo/data/ definitivo_so_Runge-Kutta_0.0080_-1.000_10.000_1.000_4.000_4.600_0.250_8.200_13.000 SO $1;