#!/bin/bash

python3 detect_events_from_spikes.py $1/$2_spikes.asc
python3 concatenate_events.py $1/ $2_spikes