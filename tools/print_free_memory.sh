#! /bin/bash

if [[ $# -lt 1  ]]; then
	echo "Usage: ./print_free_memory <time_interval (s)>"
	exit
fi
interval=$1
fout="memory_footprint.txt"
:> $fout
while true; do
	echo -n "$(date +%T) " | tee -a $fout
	free  | awk 'NR==2{printf "%.2f %.2f%%\n", $3, $3*100/$2 }' | tee -a $fout
	sleep $interval
done
