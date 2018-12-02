#! /bin/bash
# CPU affinity for icc compiler
export KMP_AFFINITY="noverbose,granularity=fine,compact,1,0"

fout=output.txt
bin=./pll_vs_pado.sh
:> $fout

$bin /scratch/ssd0/zpeng/dblp/dblp 2>&1 | tee -a $fout
$bin /scratch/ssd0/zpeng/wikitalk/wikitalk 2>&1 | tee -a $fout
$bin /scratch/ssd0/zpeng/skitter/skitter 2>&1 | tee -a $fout
#$bin /scratch/ssd0/zpeng/hollywood/hollywood 2>&1 | tee -a $fout
#$bin /scratch/ssd0/zpeng/indochina/indochina 2>&1 | tee -a $fout
#$bin /scratch/ssd0/zpeng/rmat27/rmat27 2>&1 | tee -a $fout
