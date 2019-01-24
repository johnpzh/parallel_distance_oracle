#! /bin/bash

make
./pado /scratch/ssd0/zpeng/dblp/7_w_dblp
./pado /scratch/ssd0/zpeng/wikitalk/7_w_wikitalk
./pado /scratch/ssd0/zpeng/skitter/7_w_skitter


#make
#script=tools/graph.py
#data_path=/scratch/ssd0/zpeng
#set -x
#python3 $script ${data_path}/dblp/dblp ${data_path}/dblp/7_w_dblp 1 7
#python3 $script ${data_path}/wikitalk/wikitalk ${data_path}/wikitalk/7_w_wikitalk 1 7
#python3 $script ${data_path}/skitter/skitter ${data_path}/skitter/7_w_skitter 1 7
#python3 $script ${data_path}/hollywood/hollywood ${data_path}/hollywood/7_w_hollywood 1 7
#python3 $script ${data_path}/indochina/indochina ${data_path}/indochina/7_w_indochina 1 7
#set +x

#./pado /scratch/ssd0/zpeng/dblp/dblp < tools/output.query.dblp.in > output0.txt
#./pado /scratch/ssd0/zpeng/dblp/dblp < tools/output.query.dblp.in > output1.txt
#diff output0.txt output1.txt
#gedit output0.txt

#cd tools
#./run.sh
#cd /home/zpeng/pppp/pado_seq_no_bp/tools
#./run.sh

#./pado /scratch/ssd0/zpeng/dblp/dblp 2>&1 | tee output.txt
#cd /home/zpeng/pppp/pado_seq_no_bp
#./pado /scratch/ssd0/zpeng/dblp/dblp 2>&1 | tee output.txt
