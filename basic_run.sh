#! /bin/bash

DATASETS="
gnutella
slashdot
dblp
notredame
wikitalk
youtube
trecwt10g
skitter
catdog
flickr
hollywood
indochina
uk-2002
friendster"

path=/scratch/ssd0/zpeng/collections
fout=output.txt
:> $fout

for dataset in $DATASETS; do
	echo $dataset | tee -a $fout
	echo "------" | tee -a $fout
	# unweighted, non-vectorization, non-multithread
	#./pado ${path}/${dataset}/${dataset}.txt -w 0 -v 0 -p 0 2>&1 | tee -a $fout

	# weighted, vectorized, non-multithread
	./pado ${path}/${dataset}/w_7_unif_${dataset}.txt -w 1 -v 1 -p 0 2>&1 | tee -a $fout
	echo "" | tee -a $fout
done
