#!/bin/bash

python3 ./animation/plot_inv_lymnaea.py -fs $1/$2.asc -fe $1/$2_spikes/events.asc -cu 1 -fr 400000 -sa ~/Videos/$2_inv
# python3 ./animation/plot_inv_lymnaea.py -fs $1/$2.asc -fe $1/$2_spikes/events.asc -cu 1 -fr 400000
