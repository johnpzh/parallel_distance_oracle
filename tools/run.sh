#! /bin/bash
# CPU affinity for icc compiler
export KMP_AFFINITY="noverbose,granularity=fine,compact,1,0"

fout=output.txt
bin=./pll_vs_pado.sh
:> $fout

$bin /scratch/ssd0/zpeng/dblp/1_w_dblp 2>&1 | tee -a $fout
$bin /scratch/ssd0/zpeng/wikitalk/1_w_wikitalk 2>&1 | tee -a $fout
$bin /scratch/ssd0/zpeng/skitter/1_w_skitter 2>&1 | tee -a $fout
$bin /scratch/ssd0/zpeng/hollywood/1_w_hollywood 2>&1 | tee -a $fout
$bin /scratch/ssd0/zpeng/indochina/1_w_indochina 2>&1 | tee -a $fout
#$bin /scratch/ssd0/zpeng/rmat27/rmat27 2>&1 | tee -a $fout
