#!/bin/bash

make stream

runtime1=0
runtime2=0
runtime3=0
runtime4=0
runtime5=0
runtime6=0
runtime7=0
runtime8=0
runtime9=0
runtime10=0

experiment_count=50


for (( i=1; i<=$experiment_count; i++ ))
do

	echo $i
	output=`./image_processing $1 | tail -n 1`
	runtime1=`awk -v v1=$output -v v2=$runtime1 'BEGIN{ print(v1+v2) }'`
#	echo $runtime1
	output=`./image_processing $1 $1  | tail -n 1`
	runtime2=`awk -v v1=$output -v v2=$runtime2 'BEGIN{ print(v1+v2) }'`
#	echo $runtime2
	output=`./image_processing $1 $1 $1  | tail -n 1`
	runtime3=`awk -v v1=$output -v v2=$runtime3 'BEGIN{ print(v1+v2) }'`
#	echo $runtime3
	output=`./image_processing $1 $1 $1 $1 | tail -n 1`
	runtime4=`awk -v v1=$output -v v2=$runtime4 'BEGIN{ print(v1+v2) }'`
#	echo $runtime4
	output=`./image_processing $1 $1 $1 $1 $1 | tail -n 1`
	runtime5=`awk -v v1=$output -v v2=$runtime5 'BEGIN{ print(v1+v2) }'`
#	echo $runtime5
	output=`./image_processing $1 $1 $1 $1 $1 $1 | tail -n 1`
	runtime6=`awk -v v1=$output -v v2=$runtime6 'BEGIN{ print(v1+v2) }'`
#	echo $runtime6
	output=`./image_processing $1 $1 $1 $1 $1 $1 $1 | tail -n 1`
	runtime7=`awk -v v1=$output -v v2=$runtime7 'BEGIN{ print(v1+v2) }'`
#	echo $runtime7
	output=`./image_processing $1 $1 $1 $1 $1 $1 $1 $1 | tail -n 1`
	runtime8=`awk -v v1=$output -v v2=$runtime8 'BEGIN{ print(v1+v2) }'`
#	echo $runtime8
	output=`./image_processing $1 $1 $1 $1 $1 $1 $1 $1 $1 | tail -n 1`
	runtime9=`awk -v v1=$output -v v2=$runtime9 'BEGIN{ print(v1+v2) }'`
#	echo $runtime9
	output=`./image_processing $1 $1 $1 $1 $1 $1 $1 $1 $1 $1 | tail -n 1`
	runtime10=`awk -v v1=$output -v v2=$runtime10 'BEGIN{ print(v1+v2) }'`
#	echo $runtime10

done

echo "1 Stream"
awk -v v1=$runtime1 -v v2=$experiment_count 'BEGIN{ printf "%.3f\n", v1/v2 }'
echo "2 Streams"
awk -v v1=$runtime2 -v v2=$experiment_count 'BEGIN{ printf "%.3f\n", v1/v2 }'
echo "3 Streams"
awk -v v1=$runtime3 -v v2=$experiment_count 'BEGIN{ printf "%.3f\n", v1/v2 }'
echo "4 Streams"
awk -v v1=$runtime4 -v v2=$experiment_count 'BEGIN{ printf "%.3f\n", v1/v2 }'
echo "5 Streams"
awk -v v1=$runtime5 -v v2=$experiment_count 'BEGIN{ printf "%.3f\n", v1/v2 }'
echo "6 Streams"
awk -v v1=$runtime6 -v v2=$experiment_count 'BEGIN{ printf "%.3f\n", v1/v2 }'
echo "7 Streams"
awk -v v1=$runtime7 -v v2=$experiment_count 'BEGIN{ printf "%.3f\n", v1/v2 }'
echo "8 Streams"
awk -v v1=$runtime8 -v v2=$experiment_count 'BEGIN{ printf "%.3f\n", v1/v2 }'
echo "9 Streams"
awk -v v1=$runtime9 -v v2=$experiment_count 'BEGIN{ printf "%.3f\n", v1/v2 }'
echo "10 Streams"
awk -v v1=$runtime10 -v v2=$experiment_count 'BEGIN{ printf "%.3f\n", v1/v2 }'
