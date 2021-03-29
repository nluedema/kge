#!/bin/bash

prefix=$1

echo "Deleting all but the last and best checkpoint from the following directories (prefix "$1"):"
for d in */; 
do
	if [[ $d = *$1* ]] 
	then
		cd $d
		echo $PWD
		for D in `for F in 00*; do ls -t $F/checkpoint_0*pt | tail -n 3; done`;do rm $D; done 	 # use this to actually remove
		#for D in `for F in 00*; do ls -t $F/checkpoint_0*pt | tail -n 3; done`;do echo $D; done  # use this to test
		cd ..
	fi
done
echo "Done"

