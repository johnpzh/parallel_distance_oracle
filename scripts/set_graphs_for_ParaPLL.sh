#! /bin/bash

if [[ $# -lt 1  ]]; then
	echo "Usage: ./set_graphs_forParaPLL <data_collection_path_only>"
	exit
fi
path=$1
fout=output.txt
DATASETS="
gnutella
slashdot
dblp
notredame
wikitalk
youtube"
#trecwt10g
#skitter
#catdog
#flickr
#hollywood
#indochina"
#uk-2002
#friendster"

# gnutella
data=gnutella
echo $dataset | tee -a $fout
echo "------" | tee -a $fout
cd ${path}/${dataset}
if [[ ! -f head_w_7_unif_${dataset}.txt ]]; then
	cp w_7_unif_${dataset}.txt head_w_7_unif_${dataset}.txt
	sed -i '1s/^/62586 147892\n/' head_w_7_unif_${dataset}.txt
fi

# slashdot
data=slashdot
echo $dataset | tee -a $fout
echo "------" | tee -a $fout
cd ${path}/${dataset}
if [[ ! -f head_w_7_unif_${dataset}.txt ]]; then
	cp w_7_unif_${dataset}.txt head_w_7_unif_${dataset}.txt
	sed -i '1s/^/82168 948464\n/' head_w_7_unif_${dataset}.txt
fi

# dblp
data=dblp
echo $dataset | tee -a $fout
echo "------" | tee -a $fout
cd ${path}/${dataset}
if [[ ! -f head_w_7_unif_${dataset}.txt ]]; then
	cp w_7_unif_${dataset}.txt head_w_7_unif_${dataset}.txt
	sed -i '1s/^/317080 1049866\n/' head_w_7_unif_${dataset}.txt
fi

# notredame
data=notredame
echo $dataset | tee -a $fout
echo "------" | tee -a $fout
cd ${path}/${dataset}
if [[ ! -f head_w_7_unif_${dataset}.txt ]]; then
	cp w_7_unif_${dataset}.txt head_w_7_unif_${dataset}.txt
	sed -i '1s/^/325729 1497134\n/' head_w_7_unif_${dataset}.txt
fi

# wikitalk
data=wikitalk
echo $dataset | tee -a $fout
echo "------" | tee -a $fout
cd ${path}/${dataset}
if [[ ! -f head_w_7_unif_${dataset}.txt ]]; then
	cp w_7_unif_${dataset}.txt head_w_7_unif_${dataset}.txt
	sed -i '1s/^/2394385 5021410\n/' head_w_7_unif_${dataset}.txt
fi

# youtube
data=youtube
echo $dataset | tee -a $fout
echo "------" | tee -a $fout
cd ${path}/${dataset}
if [[ ! -f head_w_7_unif_${dataset}.txt ]]; then
	cp w_7_unif_${dataset}.txt head_w_7_unif_${dataset}.txt
	sed -i '1s/^/3223589 9375374\n/' head_w_7_unif_${dataset}.txt
fi

# trecwt10g
data=trecwt10g
echo $dataset | tee -a $fout
echo "------" | tee -a $fout
cd ${path}/${dataset}
if [[ ! -f head_w_7_unif_${dataset}.txt ]]; then
	cp w_7_unif_${dataset}.txt head_w_7_unif_${dataset}.txt
	sed -i '1s/^/1601787 8063026\n/' head_w_7_unif_${dataset}.txt
fi

# skitter
data=skitter
echo $dataset | tee -a $fout
echo "------" | tee -a $fout
cd ${path}/${dataset}
if [[ ! -f head_w_7_unif_${dataset}.txt ]]; then
	cp w_7_unif_${dataset}.txt head_w_7_unif_${dataset}.txt
	sed -i '1s/^/1696415 11095298\n/' head_w_7_unif_${dataset}.txt
fi

# catdog
data=catdog
echo $dataset | tee -a $fout
echo "------" | tee -a $fout
cd ${path}/${dataset}
if [[ ! -f head_w_7_unif_${dataset}.txt ]]; then
	cp w_7_unif_${dataset}.txt head_w_7_unif_${dataset}.txt
	sed -i '1s/^/623766 15699276\n/' head_w_7_unif_${dataset}.txt
fi

# flickr
data=flickr
echo $dataset | tee -a $fout
echo "------" | tee -a $fout
cd ${path}/${dataset}
if [[ ! -f head_w_7_unif_${dataset}.txt ]]; then
	cp w_7_unif_${dataset}.txt head_w_7_unif_${dataset}.txt
	sed -i '1s/^/2302925 33140017\n/' head_w_7_unif_${dataset}.txt
fi

# hollywood
data=hollywood
echo $dataset | tee -a $fout
echo "------" | tee -a $fout
cd ${path}/${dataset}
if [[ ! -f head_w_7_unif_${dataset}.txt ]]; then
	cp w_7_unif_${dataset}.txt head_w_7_unif_${dataset}.txt
	sed -i '1s/^/1139905 113891327\n/' head_w_7_unif_${dataset}.txt
fi

# indochina
data=indochina
echo $dataset | tee -a $fout
echo "------" | tee -a $fout
cd ${path}/${dataset}
if [[ ! -f head_w_7_unif_${dataset}.txt ]]; then
	cp w_7_unif_${dataset}.txt head_w_7_unif_${dataset}.txt
	sed -i '1s/^/7414866 194109311\n/' head_w_7_unif_${dataset}.txt
fi
