#!/bin/bash

echo "Analysing $1$2"
python3 "detect_events_from_spikes.py" "$1$2.asc"
# python3 "check_events.py" "$1"
python3 "burst_charact.py" "$1" "$2" "1000"

