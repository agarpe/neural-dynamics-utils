#!/bin/bash

echo "Analysing $1"
python3 "detect_events.py" "$1"
python3 "check_events.py" "$1"

