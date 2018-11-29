#! /bin/bash

#make
#./pado /scratch/ssd0/zpeng/dblp/dblp < tools/output.query.dblp.in > output0.txt
#./pado /scratch/ssd0/zpeng/dblp/dblp < tools/output.query.dblp.in > output1.txt
#diff output0.txt output1.txt
#gedit output0.txt

cd tools
./run.sh
cd /home/zpeng/pppp/pado_seq_no_bp/tools
./run.sh

#./pado /scratch/ssd0/zpeng/dblp/dblp 2>&1 | tee output.txt
#cd /home/zpeng/pppp/pado_seq_no_bp
#./pado /scratch/ssd0/zpeng/dblp/dblp 2>&1 | tee output.txt
