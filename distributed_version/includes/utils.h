//
// Created by Zhen "John" Peng on 2019-07-21.
//

#ifndef PADO_UTILS_H
#define PADO_UTILS_H
#include <iostream>
#include <fstream>
#include <unistd.h>

namespace PADO {

class Utils {
public:

    // Function: get virtual memory and resident memory in MB
    // reference: https://gist.github.com/thirdwing/da4621eb163a886a03c5
    //            http://man7.org/linux/man-pages/man5/proc.5.html
    //            http://man7.org/linux/man-pages/man3/sysconf.3.html
    static int memory_usage(double &virt, double &res)
    {
        std::ifstream proc_ifs("/proc/self/stat");
        if (!proc_ifs.is_open()) {
            virt = 0.0;
            res = 0.0;
            return 1;
        }
        std::string ignore;
        uint64_t vm;
        uint64_t rsm;
        proc_ifs >> ignore >> ignore >> ignore >> ignore
                >> ignore >> ignore >> ignore >> ignore
                >> ignore >> ignore >> ignore >> ignore
                >> ignore >> ignore >> ignore >> ignore
                >> ignore >> ignore >> ignore >> ignore
                >> ignore >> ignore >> vm >> rsm;
        virt = vm / 1024.0 / 1024.0;
        res = rsm * (sysconf(_SC_PAGESIZE) / 1024.0 / 1024.0);

        return 0;
    }

    // Function: get total memory and free memory in MB
    // reference: https://www.cyberciti.biz/faq/ram-size-linux/
    //            https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/commit/?id=34e431b0ae398fc54ea69ff85ec700722c9da773
    static int system_memory(double &memtotal, double &memfree)
    {
        std::ifstream meminfo_ifs("/proc/meminfo");
        if (!meminfo_ifs.is_open()) {
            memtotal = 0.0;
            memfree = 0.0;
            return 1;
        }
        std::string ignore;
        uint64_t mt;
        uint64_t mf;
        meminfo_ifs >> ignore >> mt >> ignore
                    >> ignore >> mf;
        memtotal = mt / 1024.0;
        memfree = mf / 1024.0;
        return 0;
    }
};
}

#endif //PADO_UTILS_H
