#! /bin/bash

# CPU affinity for icc compiler
export KMP_AFFINITY="noverbose,granularity=fine,compact,1,0"

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

path=/scratch/ssd0/zpeng/collections
## PADO_unw_unv_unp
#report_dir="report_$(date +%Y%m%d-%H%M%S)_tile-size-${i}_memory"
#amplxe-cl -collect memory-access -result-dir ${report_dir} -data-limit=0 -knob analyze-mem-objects=true -knob analyze-openmp=true -- ./pado ${path}/${dataset}/${dataset}.txt index.index -w 0 -v 0 -p 0

cd ..
# PADO_unw_unv_unp
for dataset in $DATASETS; do
	report_dir="report_$(date +%Y%m%d-%H%M%S)_pado_w_unv_unp_${dataset}"
	amplxe-cl -collect memory-access -result-dir ${report_dir} -data-limit=0 -knob analyze-mem-objects=true -knob analyze-openmp=true -- ./pado ${path}/${dataset}/w_7_unif_${dataset}.txt index -w 1 -v 0 -p 0
done

# PADO_unw_unv_unp
for dataset in $DATASETS; do
	report_dir="report_$(date +%Y%m%d-%H%M%S)_pado_w_v_unp_${dataset}"
	amplxe-cl -collect memory-access -result-dir ${report_dir} -data-limit=0 -knob analyze-mem-objects=true -knob analyze-openmp=true -- ./pado ${path}/${dataset}/w_7_unif_${dataset}.txt index -w 1 -v 1 -p 0
done
