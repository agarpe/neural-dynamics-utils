import h5py
import sys
filename = '../24-10-2019/exp1.h5'

###############################################################
## Este programa realiza una exploración de un hd5,
## devolviendo todos los posibles group y dataset
## Esta pensado para utilizarse de la siguiente manera:
## - Se localiza el dataset requerido
## - Se localiza en que group esta ubicado
## - Con esto ya puede ser cargada la traza
###############################################################

###############################################################
## Al abrid un hdf5 este nos devuelve un diccionario
## Cada elemento alamacenado puede ser:
## a) HDF5 group (podemos considerar la raiz como uno de estos)
##    Para los grupos se nos indican su numero de miembros
## b) HDF5 dataset
##    Datos en si
###############################################################

######################################################
## Si queremos entrar en un group debemos contar con:
## - El group anterior para acceder
## - Conocer su nombre o el indice en el group padre 
######################################################

###############################################################
## Para cada grupo imprimimos sus dataset
## Tras esto se exploran recursivamente sus group
###############################################################
def explore_group(f, key, value, p):
	print ('\n\033[91m', p, value, '\033[0m')
	# Exploramos datasets
	for elem in value:
		if 'HDF5 dataset' in str(f[str(key)+'/'+str(elem)]):
			print (p+'      "'+str(f[str(key)+'/'+str(elem)]))
	# Continuamos a subgrupos
	for elem in value:
		if 'HDF5 group' in str(f[str(key)+'/'+str(elem)]):
			key_new = key+'/'+elem
			explore_group(f, key_new, f[key_new], p+'    ')

###############################################################
## La raiz de un HDF5 es de tipo HDF5
## *creo* que solo debería tener groups
###############################################################

##File as argument
if len(sys.argv) > 2:
	path = sys.argv[1]
	file = sys.argv[2]
	filename = path + file
else:
	path = ''
	filename  = 'exp1.h5'


f = h5py.File(filename, 'r')
print('\n'+str(f))
for elem in f:
	p = '    '
	explore_group(f, elem, f[elem], p)
print()
