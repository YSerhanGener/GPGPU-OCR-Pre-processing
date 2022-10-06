#!/bin/bash

runtime=0
LD_LIBRARY_PATH=$HOME/local/lib/:$LD_LIBRARY_PATH

experiment_count=1000

make

for (( i=1; i<=$experiment_count; i++ ))
do
	#echo "Running experiement " $i " ./lep_tes ../inputs/4_GS8homeScreen.jpg"
	output=`./lep_tes ../inputs/4_GS8homeScreen.jpg | head -n 1 | tr ':' '\n' | tail -n 1 | tr ' ' '\n' | head -n 1`
	runtime=`awk -v v1=$output -v v2=$runtime 'BEGIN{ print(v1+v2) }'`
	#echo $output
	#echo $runtime
done
awk -v v1=$runtime -v v2=$experiment_count 'BEGIN{ printf "%.3f\n", v1/v2 }'
