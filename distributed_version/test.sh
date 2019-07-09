#! /bin/bash

output=output.txt

set -x
cd /home/zpeng/pppp/pado/cmake-build-debug-intel-compiler/distributed_version
make
for ((n = 1; n < 17; n *= 2)); do
	echo "-------  Number_of_Hosts: ${n} -------" >> ${output}
	mpiexec -n ${n} ./dpado ~/scratch/dblp/binary.dblp  2>&1 | tee -a ${output}
done
set +x
