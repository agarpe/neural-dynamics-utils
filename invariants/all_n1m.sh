#!/bin/bash

python3 burst_charact.py ../articulo/data/ definitivo_n1m_Euler_0.0100_8.500_-1.000_2.000_0.000_4.600_0.500_0.000_10.500 N1M $1;
python3 burst_charact.py ../articulo/data/ definitivo_n1m_Euler_0.0010_8.500_-1.000_2.000_0.000_4.600_0.500_0.000_10.500 N1M $1;
python3 burst_charact.py ../articulo/data/ definitivo_n1m_Runge-Kutta_0.0050_8.500_-1.000_2.000_0.000_4.600_0.500_0.000_10.500 N1M $1;
python3 burst_charact.py ../articulo/data/ definitivo_n1m_Runge-Kutta_0.0080_8.500_-1.000_2.000_0.000_4.600_0.500_0.000_10.500 N1M $1;