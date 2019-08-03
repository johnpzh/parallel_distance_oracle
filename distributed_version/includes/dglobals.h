//
// Created by Zhen Peng on 5/14/19.
//

#ifndef PADO_DGLOBALS_H
#define PADO_DGLOBALS_H

#include <stdlib.h>
#include <limits.h>
#include <sys/stat.h>
#include <assert.h>

namespace PADO {

typedef uint32_t VertexID;
typedef uint64_t EdgeID;
typedef uint8_t UnweightedDist;
const UnweightedDist MAX_UNWEIGHTED_DIST = UCHAR_MAX;

// Get the file size
unsigned long get_file_size(const char *filename)
{
    struct stat file_stat;
    assert(stat(filename, &file_stat) == 0);
    return file_stat.st_size;
}



} // End namespace PADO

#endif //PADO_DGLOBALS_H
