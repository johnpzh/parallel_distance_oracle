#! /bin/bash

if [[ $# -lt 1 ]]; then
	echo "Usage: ./pll_vs_pado.sh <input_data>"
	exit
fi
fin=$1
pado="../pado"
pll="/home/zpeng/pppp/pll_dijkstra_simd/bin/construct_index"

echo $fin
echo "--------------------------------------------------"
echo "PADO:"
echo "-----"
$pado $fin
#echo ""
#
#echo "PLL:"
#echo "----"
#$pll $fin pll.label
echo "--------------------------------------------------"
##rm pll.label
