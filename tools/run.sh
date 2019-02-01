#! /bin/bash
# CPU affinity for icc compiler
export KMP_AFFINITY="noverbose,granularity=fine,compact,1,0"

fout=output.txt
bin=./pll_vs_pado.sh
:> $fout

# Weighted Graphs
$bin /scratch/ssd0/zpeng/dblp/7_w_dblp 2>&1 | tee -a $fout
$bin /scratch/ssd0/zpeng/wikitalk/7_w_wikitalk 2>&1 | tee -a $fout
$bin /scratch/ssd0/zpeng/skitter/7_w_skitter 2>&1 | tee -a $fout
$bin /scratch/ssd0/zpeng/hollywood/7_w_hollywood 2>&1 | tee -a $fout
$bin /scratch/ssd0/zpeng/indochina/7_w_indochina 2>&1 | tee -a $fout
#$bin /scratch/ssd0/zpeng/rmat27/rmat27 2>&1 | tee -a $fout

# Unweighted Graphs
#$bin /scratch/ssd0/zpeng/dblp/dblp 2>&1 | tee -a $fout
#$bin /scratch/ssd0/zpeng/wikitalk/wikitalk 2>&1 | tee -a $fout
#$bin /scratch/ssd0/zpeng/skitter/skitter 2>&1 | tee -a $fout
#$bin /scratch/ssd0/zpeng/hollywood/hollywood 2>&1 | tee -a $fout
#$bin /scratch/ssd0/zpeng/indochina/indochina 2>&1 | tee -a $fout
#$bin /scratch/ssd0/zpeng/rmat27/rmat27 2>&1 | tee -a $fout
