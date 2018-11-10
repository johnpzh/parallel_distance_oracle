#! /bin/bash

unset OMP_PLACES
unset OMP_PROC_BIND
unset KMP_AFFINITY
./run.sh
cp output.txt output.scalability_no_aff.txt

export KMP_AFFINITY="verbose,granularity=fine,compact,1,0"
./run.sh
cp output.txt output.scalability_set_aff.txt
