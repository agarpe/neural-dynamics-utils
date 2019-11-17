#!/bin/bash
for f in $1*.dat; do 
    mv -- "$f" "${f%.dat}.asc"
done
