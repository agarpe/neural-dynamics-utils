#!/bin/bash

python3 ./animation/plot_inv_lymnaea.py -fs $1/$2.asc -fe $1/$2_spikes/events.asc -mo 1 -cu 1 -fr 400000 -sa $2_mo1_current
