#! /bin/bash


set -x
cd /home/zpeng/pppp/clion/pado_th107a2/cmake-build-debug/distributed_version
make
for ((n = 2; n < 17; n *= 2)); do
	echo "-------  Number_of_Hosts: ${n} -------"
	mpiexec -np ${n} -genv I_MPI_DEBUG +3 ./dpado ~/scratch/indochina/indochina.binary  2>&1 | tee "output${n}.txt"
done
set +x
