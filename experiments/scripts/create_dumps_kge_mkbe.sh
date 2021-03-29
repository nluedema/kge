#!/bin/sh

keys=$1

for d in */ ; do
	if [[ $d = *"-mkbe-"* ]]
	then
		cd $d
		#echo $d
		kge dump trace . --keysfile $keys > trace_dump.csv
		cd ..
	fi
done
