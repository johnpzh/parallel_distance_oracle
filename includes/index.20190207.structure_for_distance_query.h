/*
 * index.h
 *
 *  Created on: Feb 7, 2019
 *      Author: Zhen Peng
 */

#ifndef INCLUDES_INDEX_H_
#define INCLUDES_INDEX_H_


#include <cstdlib>
#include <vector>
#include <queue>
#include "globals.h"

namespace PADO {

extern inti BITPARALLEL_SIZE;

struct IndexOrdered {
	smalli bp_dist[BITPARALLEL_SIZE];
	uint64_t bp_sets[BITPARALLEL_SIZE][2]; // [0]: S^{-1}, [1]: S^{0}

	vector<idi> label_id;
	vector<weighti> label_dists;
};

}


#endif /* INCLUDES_INDEX_H_ */
