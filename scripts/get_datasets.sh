#! /bin/bash

if [[ $# -lt 1 ]]; then
	echo "Usage: ./get_datasets.sh <output_directory>"
	exit
fi
dout=$1
mkdir -p $dout

# Gnutella
cd $dout
dataname=gnutella
mkdir -p $dataname
cd $dataname
if [[ ! -f ${dataname}.txt ]]; then
	wget -c https://snap.stanford.edu/data/p2p-Gnutella31.txt.gz && gunzip -k p2p-Gnutella31.txt.gz
	mv p2p-Gnutella31.txt ${dataname}.txt
	sed -i '/^[#%]/ d' ${dataname}.txt
fi

# Slashdot
cd $dout
dataname=slashdot
mkdir -p $dataname
cd $dataname
if [[ ! -f ${dataname}.txt ]]; then
	wget -c https://snap.stanford.edu/data/soc-Slashdot0902.txt.gz && gunzip -k soc-Slashdot0902.txt.gz
	mv soc-Slashdot0902.txt ${dataname}.txt
	sed -i '/^[#%]/ d' ${dataname}.txt
fi

# DBLP
cd $dout
dataname=dblp
mkdir -p $dataname
cd $dataname
if [[ ! -f ${dataname}.txt ]]; then
	wget -c http://konect.uni-koblenz.de/downloads/tsv/com-dblp.tar.bz2 && tar xjvf com-dblp.tar.bz2
	mv com-dblp/out.com-dblp ${dataname}.txt
	sed -i '/^[#%]/ d' ${dataname}.txt
fi

# Notre Dame
cd $dout
dataname=notredame
mkdir -p $dataname
cd $dataname
if [[ ! -f ${dataname}.txt ]]; then
	wget -c https://snap.stanford.edu/data/web-NotreDame.txt.gz && gunzip -k web-NotreDame.txt.gz
	mv web-NotreDame.txt ${dataname}.txt
	sed -i '/^[#%]/ d' ${dataname}.txt
fi

# WikiTalk
cd $dout
dataname=wikitalk
mkdir -p $dataname
cd $dataname
if [[ ! -f ${dataname}.txt ]]; then
	wget -c https://snap.stanford.edu/data/wiki-Talk.txt.gz && gunzip -k wiki-Talk.txt.gz
	mv wiki-Talk.txt ${dataname}.txt
	sed -i '/^[#%]/ d' ${dataname}.txt
fi

# Youtube
cd $dout
dataname=youtube
mkdir -p $dataname
cd $dataname
if [[ ! -f ${dataname}.txt ]]; then
	wget -c http://konect.uni-koblenz.de/downloads/tsv/youtube-u-growth.tar.bz2 && tar xjvf youtube-u-growth.tar.bz2
	mv youtube-u-growth/out.youtube-u-growth ${dataname}.txt
	sed '/^[#%]/ d' ${dataname}.txt > ${dataname}.tmp
	cut -d ' ' -f 1,2 < ${dataname}.tmp > ${dataname}.txt
	rm -f ${dataname}.tmp
fi

# TREC WT10g
cd $dout
dataname=trecwt10g
mkdir -p $dataname
cd $dataname
if [[ ! -f ${dataname}.txt ]]; then
	wget -c http://konect.uni-koblenz.de/downloads/tsv/trec-wt10g.tar.bz2 && tar xjvf trec-wt10g.tar.bz2
	mv trec-wt10g/out.trec-wt10g ${dataname}.txt
	sed -i '/^[#%]/ d' ${dataname}.txt
fi

# Skitter
cd $dout
dataname=skitter
mkdir -p $dataname
cd $dataname
if [[ ! -f ${dataname}.txt ]]; then
	wget -c http://konect.uni-koblenz.de/downloads/tsv/as-skitter.tar.bz2 && tar xjvf as-skitter.tar.bz2
	mv as-skitter/out.as-skitter ${dataname}.txt
	sed -i '/^[#%]/ d' ${dataname}.txt
fi

# Catster/Dogster Familylinks/Friendships
cd $dout
dataname=catdog
mkdir -p $dataname
cd $dataname
if [[ ! -f ${dataname}.txt ]]; then
	wget -c http://konect.uni-koblenz.de/downloads/tsv/petster-carnivore.tar.bz2 && tar xjvf petster-carnivore.tar.bz2
	mv petster-carnivore/out.petster-carnivore ${dataname}.txt
	sed -i '/^[#%]/ d' ${dataname}.txt
fi

# Flickr
cd $dout
dataname=flickr
mkdir -p $dataname
cd $dataname
if [[ ! -f ${dataname}.txt ]]; then
	wget -c http://konect.uni-koblenz.de/downloads/tsv/flickr-growth.tar.bz2 && tar xjvf flickr-growth.tar.bz2
	mv flickr-growth/out.flickr-growth ${dataname}.txt
	sed '/^[#%]/ d' ${dataname}.txt > ${dataname}.tmp
	cut -d ' ' -f 1,2 < ${dataname}.tmp > ${dataname}.txt
	rm -f ${dataname}.tmp
fi

# Hollywood
cd $dout
dataname=hollywood
mkdir -p $dataname
cd $dataname
if [[ ! -f ${dataname}.txt ]]; then
	wget -c https://sparse.tamu.edu/MM/LAW/hollywood-2009.tar.gz && tar xzvf hollywood-2009.tar.gz
	mv hollywood-2009/hollywood-2009.mtx ${dataname}.txt
	sed -r -i -e '/^[#%]/ d' -e '/^([0-9]+) ([0-9]+) ([0-9]+)$/ d' ${dataname}.txt
fi

# Indochina
cd $dout
dataname=indochina
mkdir -p $dataname
cd $dataname
if [[ ! -f ${dataname}.txt ]]; then
	wget -c https://sparse.tamu.edu/MM/LAW/indochina-2004.tar.gz && tar xzvf indochina-2004.tar.gz
	mv indochina-2004/indochina-2004.mtx ${dataname}.txt
	sed -r -i -e '/^[#%]/ d' -e '/^([0-9]+) ([0-9]+) ([0-9]+)$/ d' ${dataname}.txt
fi

# UK-2002
cd $dout
dataname=uk-2002
mkdir -p $dataname
cd $dataname
if [[ ! -f ${dataname}.txt ]]; then
	wget -c https://sparse.tamu.edu/MM/LAW/uk-2002.tar.gz && tar xzvf uk-2002.tar.gz
	mv uk-2002/uk-2002.mtx ${dataname}.txt
	sed -r -i -e '/^[#%]/ d' -e '/^([0-9]+) ([0-9]+) ([0-9]+)$/ d' ${dataname}.txt
fi

# Friendster
cd $dout
dataname=friendster
mkdir -p $dataname
cd $dataname
if [[ ! -f ${dataname}.txt ]]; then
	wget -c https://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz && gunzip -k com-friendster.ungraph.txt.gz
	mv com-friendster.ungraph.txt ${dataname}.txt
	sed -i '/^[#%]/ d' ${dataname}.txt
fi

# Road Graph
# BAY
cd $dout
dataname=road_bay
mkdir -p $dataname
cd $dataname
if [[ ! -f ${dataname}.txt ]]; then
	wget -c http://www.dis.uniroma1.it/challenge9/data/USA-road-d/USA-road-d.BAY.gr.gz && gunzip -k USA-road-d.BAY.gr.gz
	mv USA-road-d.BAY.gr ${dataname}.txt
	sed '/^[cp]/ d' ${dataname}.txt > ${dataname}.tmp
	cut -d ' ' -f 2,3 < ${dataname}.tmp > ${dataname}.txt
	sed -i 'n; d' ${dataname}.txt
	rm ${dataname}.tmp
fi

# FLA
cd $dout
dataname=road_fla
mkdir -p $dataname
cd $dataname
if [[ ! -f ${dataname}.txt ]]; then
	wget -c http://www.dis.uniroma1.it/challenge9/data/USA-road-d/USA-road-d.FLA.gr.gz && gunzip -k USA-road-d.FLA.gr.gz
	mv USA-road-d.FLA.gr ${dataname}.txt
	sed '/^[cp]/ d' ${dataname}.txt > ${dataname}.tmp
	cut -d ' ' -f 2,3 < ${dataname}.tmp > ${dataname}.txt
	sed -i 'n; d' ${dataname}.txt
	rm ${dataname}.tmp
fi

# CAL
cd $dout
dataname=road_cal
mkdir -p $dataname
cd $dataname
if [[ ! -f ${dataname}.txt ]]; then
	wget -c http://www.dis.uniroma1.it/challenge9/data/USA-road-d/USA-road-d.CAL.gr.gz && gunzip -k USA-road-d.CAL.gr.gz
	mv USA-road-d.CAL.gr ${dataname}.txt
	sed '/^[cp]/ d' ${dataname}.txt > ${dataname}.tmp
	cut -d ' ' -f 2,3 < ${dataname}.tmp > ${dataname}.txt
	sed -i 'n; d' ${dataname}.txt
	rm ${dataname}.tmp
fi

# E
cd $dout
dataname=road_e
mkdir -p $dataname
cd $dataname
if [[ ! -f ${dataname}.txt ]]; then
	wget -c http://www.dis.uniroma1.it/challenge9/data/USA-road-d/USA-road-d.E.gr.gz && gunzip -k USA-road-d.E.gr.gz
	mv USA-road-d.E.gr ${dataname}.txt
	sed '/^[cp]/ d' ${dataname}.txt > ${dataname}.tmp
	cut -d ' ' -f 2,3 < ${dataname}.tmp > ${dataname}.txt
	sed -i 'n; d' ${dataname}.txt
	rm ${dataname}.tmp
fi

# W
cd $dout
dataname=road_w
mkdir -p $dataname
cd $dataname
if [[ ! -f ${dataname}.txt ]]; then
	wget -c http://www.dis.uniroma1.it/challenge9/data/USA-road-d/USA-road-d.W.gr.gz && gunzip -k USA-road-d.W.gr.gz
	mv USA-road-d.W.gr ${dataname}.txt
	sed '/^[cp]/ d' ${dataname}.txt > ${dataname}.tmp
	cut -d ' ' -f 2,3 < ${dataname}.tmp > ${dataname}.txt
	sed -i 'n; d' ${dataname}.txt
	rm ${dataname}.tmp
fi

# CTR
cd $dout
dataname=road_ctr
mkdir -p $dataname
cd $dataname
if [[ ! -f ${dataname}.txt ]]; then
	wget -c http://www.dis.uniroma1.it/challenge9/data/USA-road-d/USA-road-d.CTR.gr.gz && gunzip -k USA-road-d.CTR.gr.gz
	mv USA-road-d.CTR.gr ${dataname}.txt
	sed '/^[cp]/ d' ${dataname}.txt > ${dataname}.tmp
	cut -d ' ' -f 2,3 < ${dataname}.tmp > ${dataname}.txt
	sed -i 'n; d' ${dataname}.txt
	rm ${dataname}.tmp
fi

# USA
cd $dout
dataname=road_usa
mkdir -p $dataname
cd $dataname
if [[ ! -f ${dataname}.txt ]]; then
	wget -c http://www.dis.uniroma1.it/challenge9/data/USA-road-d/USA-road-d.USA.gr.gz && gunzip -k USA-road-d.USA.gr.gz
	mv USA-road-d.USA.gr ${dataname}.txt
	sed '/^[cp]/ d' ${dataname}.txt > ${dataname}.tmp
	cut -d ' ' -f 2,3 < ${dataname}.tmp > ${dataname}.txt
	sed -i 'n; d' ${dataname}.txt
	rm ${dataname}.tmp
fi
