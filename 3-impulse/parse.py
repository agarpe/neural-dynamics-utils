import numpy as np
import sys
if (len(sys.argv) > 1):
	name = sys.argv[1]


def parse_params(name):
	# print(name)
	synapse_params = name.split("-2")[1]

	synapse_params = synapse_params.split("_")[1:-1]
	# print(synapse_params)

	params = {}

	params['tau_rec'] = synapse_params[0]
	params['tau_fac'] = synapse_params[1]
	params['A_syn'] = synapse_params[2]
	params['tau_in'] = synapse_params[3]
	if(len(synapse_params)>4):
		params['rel_prob'] = synapse_params[4]

	# print(params)

	return params



# print(parse_params(name))