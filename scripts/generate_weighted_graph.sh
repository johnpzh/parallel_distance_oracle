#! /bin/bash
# Generage weighted graphs basied on those unweighted graph got from the get_datasets.sh script.
# The used output_directory should be the same one used for the get_datasets.sh script.
# The weight of edge is hard-coded as a random number uniform from 1 to 7.

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
indochina"
#uk-2002
#friendster"

if [[ $# -lt 1 ]]; then
	echo "Usage: ./generate_weighted_graph.sh <output_directory>"
	exit
fi
dout=$1

for dataset in $DATASETS; do
	echo $dataset
	datafile=${dout}/${dataset}/${dataset}.txt
	if [[ -f $datafile ]]; then
		python3 ../tools/graph.py $datafile ${dout}/${dataset}/w_7_unif_${dataset}.txt 1 7
	else
		echo "$datafile does not exist yet."
	fi
	echo Done.
	echo ""
done
