#! /bin/bash

#unset OMP_PLACES
#unset OMP_PROC_BIND
#unset KMP_AFFINITY
#./run.sh
#cp output.txt output.scalability_no_aff.txt
#
#export KMP_AFFINITY="verbose,granularity=fine,compact,1,0"
#./run.sh
#cp output.txt output.scalability_set_aff.txt


DATASETS="
gnutella
slashdot
notredame
youtube
trecwt10g
catdog
flickr
uk-2002
friendster"

path=/scratch/ssd0/zpeng/collections
for dataset in $DATASETS; do
	echo $dataset
	python3 graph.py ${path}/${dataset}/${dataset}.txt ${path}/${dataset}/w_7_unif_${dataset}.txt 1 7
	echo "Done."
done
