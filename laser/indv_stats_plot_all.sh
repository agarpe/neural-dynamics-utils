#!/bin/bash

python3 stats_plot.py ../../data/laser/27-Jul-2020/events/ | tee ../../data/laser/27-Jul-2020/events/stats.log;
python3 stats_plot.py ../../data/laser/29-Jul-2020/events/ | tee ../../data/laser/29-Jul-2020/events/stats.log;
python3 stats_plot.py ../../data/laser/09-Sep-2020/events/ | tee ../../data/laser/09-Sep-2020/events/stats.log;
python3 stats_plot.py ../../data/laser/10-Sep-2020/events/ | tee ../../data/laser/10-Sep-2020/events/stats.log;
python3 stats_plot.py ../../data/laser/23-Oct-2020/events/ | tee ../../data/laser/23-Oct-2020/events/stats.log;
python3 stats_plot.py ../../data/laser/30-Oct-2020/events/ | tee ../../data/laser/30-Oct-2020/events/stats.log;
python3 stats_plot.py ../../data/laser/04-Nov-2020/events/ | tee ../../data/laser/04-Nov-2020/events/stats.log;
python3 stats_plot.py ../../data/laser/13-Nov-2020/events/ | tee ../../data/laser/13-Nov-2020/events/stats.log;
