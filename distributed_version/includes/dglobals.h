//
// Created by Zhen Peng on 5/14/19.
//

#ifndef PADO_DGLOBALS_H
#define PADO_DGLOBALS_H

#include <stdlib.h>
#include <sys/stat.h>
#include <assert.h>

namespace PADO {

typedef uint32_t VertexID;
typedef uint64_t EdgeID;

// Get the file size
unsigned long get_file_size(char *filename)
{
    struct stat file_stat;
    assert(stat(filename, &file_stat) == 0);
    return file_stat.st_size;
}



} // End namespace PADO

#endif //PADO_DGLOBALS_H
