//
// Created by Zhen Peng on 1/17/20.
//

/**
 * Process a edge-list graph, remove all equivalent vertices, then store into a new binary graph.
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include <set>
#include <immintrin.h>
#include "dglobals.h"
#include "globals.h"
#include <omp.h>

//bool are_equivalent(
//        const PADO::VertexID a_v,
//        const PADO::VertexID b_v,
//        const std::vector<std::vector<PADO::VertexID> > &adjacency_list)
//{
//    if (adjacency_list[a_v].size() != adjacency_list[b_v].size()) {
//        return false;
//    }
//    std::set<PADO::VertexID> a_set;
//    std::set<PADO::VertexID> b_set;
//
//    // A
//    a_set.insert(a_v);
//    for (PADO::VertexID vn : adjacency_list[a_v]) {
//        a_set.insert(vn);
//    }
//
//    // B
//    b_set.insert(b_v);
//    for (PADO::VertexID vn : adjacency_list[b_v]) {
//        b_set.insert(vn);
//    }
//
//    return a_set == b_set;
//}

//void eliminate_vertices(char *input_filename, char *output_filename)
//{
//    double time_running = -PADO::WallTimer::get_time_mark();
//
//    std::ifstream fin(input_filename);
//    if (!fin.is_open()) {
//        fprintf(stderr,
//                "Error: cannot open file %s\n", input_filename);
//        exit(EXIT_FAILURE);
//    }
//
//    std::ofstream fout(output_filename);
//    if(!fout.is_open()) {
//        fprintf(stderr,
//                "Error: cannot create file %s\n", output_filename);
//        exit(EXIT_FAILURE);
//    }
//
//    std::vector<bool> is_redundant;
//    {
//        // Read the graph.
//        // Get the num_v at first.
//        PADO::VertexID num_v = 0;
//        PADO::EdgeID num_e = 0;
//        std::string line;
//        while (std::getline(fin, line)) {
//            if (line[0] == '#' || line[0] == '%' || line[0] == '+' || line[0] == '-') {
//                continue;
//            }
//            std::istringstream iss(line);
//            PADO::VertexID head;
//            PADO::VertexID tail;
//            iss >> head >> tail;
//            num_v = std::max(num_v, std::max(head, tail) + 1);
//            ++num_e;
//        }
//        printf("intput: num_v: ");
//        std::cout << num_v;
//        printf(" num_e: ");
//        std::cout << num_e << std::endl;
//
//        // Read the graph again, get the adjacency_list.
//        fin.clear();
//        fin.seekg(0);
//        std::vector<std::vector<PADO::VertexID> > adjacency_list(num_v);
//        while (std::getline(fin, line)) {
//            if (line[0] == '#' || line[0] == '%' || line[0] == '+' || line[0] == '-') {
//                continue;
//            }
//            std::istringstream iss(line);
//            PADO::VertexID head;
//            PADO::VertexID tail;
//            iss >> head >> tail;
//            adjacency_list[head].push_back(tail);
//            adjacency_list[tail].push_back(head);
//        }
//
//        // Check every vertex if it is redundant.
//        is_redundant.resize(num_v, false);
//        for (PADO::VertexID a_v = 0; a_v < num_v; ++a_v) {
//            if (is_redundant[a_v]) {
//                continue;
//            }
//            // Prepare a_v's set;
//            std::set<PADO::VertexID> a_set;
//            a_set.insert(a_v);
//            for (PADO::VertexID vn : adjacency_list[a_v]) {
//                a_set.insert(vn);
//            }
////#pragma omp parallel for
//            for (PADO::VertexID b_v = a_v + 1; b_v < num_v; ++b_v) {
//                if (is_redundant[b_v] || adjacency_list[a_v].size() != adjacency_list[b_v].size()) {
//                    continue;
//                }
//                // Check b_v itself.
//                if (a_set.find(b_v) == a_set.end()) {
//                    continue;
//                }
//                // Check b_v's neighbors.
//                bool all_equal = true;
//                for (const PADO::VertexID vn : adjacency_list[b_v]) {
////                PADO::VertexID b_outdegree = adjacency_list[b_v].size();
////                for (PADO::VertexID v_i = 0; v_i < b_outdegree; ++v_i) {
////                    PADO::VertexID vn = adjacency_list[b_v][v_i];
//                    if (a_set.find(vn) == a_set.end()) {
//                        all_equal = false;
//                        break;
//                    }
//                }
//                if (all_equal) {
//                    is_redundant[b_v] = true;
//                }
//            }
//        }
//    }
//
//    {
//        // Read the graph again, do the reduction.
//        std::vector< std::pair<PADO::VertexID, PADO::VertexID> > edge_list;
//        PADO::VertexID num_v = 0;
//        PADO::EdgeID num_e = 0;
//        fin.clear();
//        fin.seekg(0);
//        std::string line;
//        while (std::getline(fin, line)) {
//            if (line[0] == '#' || line[0] == '%' || line[0] == '+' || line[0] == '-') {
//                continue;
//            }
//            std::istringstream iss(line);
//            PADO::VertexID head;
//            PADO::VertexID tail;
//            iss >> head >> tail;
//            if (is_redundant[head] || is_redundant[tail]) {
//                continue;
//            }
//            num_v = std::max(num_v, std::max(head, tail) + 1);
//            ++num_e;
//            edge_list.emplace_back(head, tail);
//        }
//        std::cout << "output: num_v: " << num_v << " num_e: " << num_e << std::endl;
//
//        // Write into the binary file.
//        fout.write(reinterpret_cast<char *>(&num_v), sizeof(num_v));
//        fout.write(reinterpret_cast<char *>(&num_e), sizeof(num_e));
//        for (const auto &edge : edge_list) {
//            PADO::VertexID head = edge.first;
//            PADO::VertexID tail = edge.second;
//            fout.write(reinterpret_cast<char *>(&head), sizeof(head));
//            fout.write(reinterpret_cast<char *>(&tail), sizeof(tail));
//        }
//    }
//
//    time_running += PADO::WallTimer::get_time_mark();
//    printf("running_time(s.): %f\n", time_running);
//}

//// 20200118-2319
//void eliminate_vertices(char *input_filename, char *output_filename)
//{
//    double time_running = -PADO::WallTimer::get_time_mark();
//
//    std::ifstream fin(input_filename);
//    if (!fin.is_open()) {
//        fprintf(stderr,
//                "Error: cannot open file %s\n", input_filename);
//        exit(EXIT_FAILURE);
//    }
//
//    std::ofstream fout(output_filename);
//    if(!fout.is_open()) {
//        fprintf(stderr,
//                "Error: cannot create file %s\n", output_filename);
//        exit(EXIT_FAILURE);
//    }
//
//    std::vector<uint8_t> is_redundant;
//    {
//        // Read the graph.
//        // Get the num_v at first.
//        PADO::VertexID num_v = 0;
//        PADO::EdgeID num_e = 0;
//        std::string line;
//        while (std::getline(fin, line)) {
//            if (line[0] == '#' || line[0] == '%' || line[0] == '+' || line[0] == '-') {
//                continue;
//            }
//            std::istringstream iss(line);
//            PADO::VertexID head;
//            PADO::VertexID tail;
//            iss >> head >> tail;
//            num_v = std::max(num_v, std::max(head, tail) + 1);
//            ++num_e;
//        }
//        printf("intput: num_v: ");
//        std::cout << num_v;
//        printf(" num_e: ");
//        std::cout << num_e << std::endl;
//
//        // Read the graph again, get the adjacency_list.
//        fin.clear();
//        fin.seekg(0);
//        std::vector< std::set<PADO::VertexID> > adjacency_list(num_v);
//        for (PADO::VertexID v_id = 0; v_id < num_v; ++v_id) {
//            adjacency_list[v_id].insert(v_id);
//        }
//        while (std::getline(fin, line)) {
//            if (line[0] == '#' || line[0] == '%' || line[0] == '+' || line[0] == '-') {
//                continue;
//            }
//            std::istringstream iss(line);
//            PADO::VertexID head;
//            PADO::VertexID tail;
//            iss >> head >> tail;
//            adjacency_list[head].insert(tail);
//            adjacency_list[tail].insert(head);
//        }
//
//        // Check every vertex if it is redundant.
//        is_redundant.resize(num_v, 0);
//        for (PADO::VertexID a_v = 0; a_v < num_v; ++a_v) {
//            if (is_redundant[a_v]) {
//                continue;
//            }
//            const auto &a_set = adjacency_list[a_v];
//#pragma omp parallel for
//            for (PADO::VertexID b_v = a_v + 1; b_v < num_v; ++b_v) {
//                if (is_redundant[b_v]) {
//                    continue;
//                }
//                const auto &b_set = adjacency_list[b_v];
//                if (a_set.size() != b_set.size()) {
//                    continue;
//                }
//                if (a_set == b_set) {
//                    is_redundant[b_v] = 1;
//                }
//            }
//        }
//    }
//
//    printf("Writing...\n");
//    {
//        // Read the graph again, do the reduction.
//        std::vector< std::pair<PADO::VertexID, PADO::VertexID> > edge_list;
//        PADO::VertexID num_v = 0;
//        PADO::EdgeID num_e = 0;
//        fin.clear();
//        fin.seekg(0);
//        std::string line;
//        while (std::getline(fin, line)) {
//            if (line[0] == '#' || line[0] == '%' || line[0] == '+' || line[0] == '-') {
//                continue;
//            }
//            std::istringstream iss(line);
//            PADO::VertexID head;
//            PADO::VertexID tail;
//            iss >> head >> tail;
//            if (is_redundant[head] || is_redundant[tail]) {
//                continue;
//            }
//            num_v = std::max(num_v, std::max(head, tail) + 1);
//            ++num_e;
//            edge_list.emplace_back(head, tail);
//        }
//        std::cout << "output: num_v: " << num_v << " num_e: " << num_e << std::endl;
//
//        // Write into the binary file.
//        fout.write(reinterpret_cast<char *>(&num_v), sizeof(num_v));
//        fout.write(reinterpret_cast<char *>(&num_e), sizeof(num_e));
//        for (const auto &edge : edge_list) {
//            PADO::VertexID head = edge.first;
//            PADO::VertexID tail = edge.second;
//            fout.write(reinterpret_cast<char *>(&head), sizeof(head));
//            fout.write(reinterpret_cast<char *>(&tail), sizeof(tail));
//        }
//    }
//
//    time_running += PADO::WallTimer::get_time_mark();
//    printf("running_time(s.): %f\n", time_running);
//}

inline bool are_equivalent(
        const std::vector<PADO::VertexID> &a_list,
        const std::vector<PADO::VertexID> &b_list)
{
//// AVX-512 version
#if defined(__AVX512F__)
    PADO::VertexID list_size = a_list.size();
    const uint32_t CHUNK_SIZE = 16;
    PADO::VertexID remainer = list_size % CHUNK_SIZE;
    PADO::VertexID v_i_bound = list_size - remainer;

    for (PADO::VertexID v_i = 0; v_i < v_i_bound; v_i += CHUNK_SIZE) {
//        __m512i a_p = _mm512_loadu_epi32(a_list.data() + v_i);
//        __m512i b_p = _mm512_loadu_epi32(b_list.data() + v_i);
        __m512i a_p = _mm512_loadu_si512(a_list.data() + v_i);
        __m512i b_p = _mm512_loadu_si512(b_list.data() + v_i);
        if (_mm512_cmpneq_epi32_mask(a_p, b_p)) {
            return false;
        }
    }
    if (remainer) {
        for (PADO::VertexID v_i = v_i_bound; v_i < list_size; ++v_i) {
            if (a_list[v_i] != b_list[v_i]) {
                return false;
            }
        }
    }

    return true;

//// AVX2 Version
#elif defined(__AVX2__)
    PADO::VertexID list_size = a_list.size();
    const uint32_t CHUNK_SIZE = 8;
    PADO::VertexID remainer = list_size % CHUNK_SIZE;
    PADO::VertexID v_i_bound = list_size - remainer;

    for (PADO::VertexID v_i = 0; v_i < v_i_bound; v_i += CHUNK_SIZE) {
        __m256i a_p = _mm256_lddqu_si256((__m256i const *) (a_list.data() + v_i));
        __m256i b_p = _mm256_lddqu_si256((__m256i const *) (b_list.data() + v_i));
//        if (_mm256_cmpneq_epi32_mask(a_p, b_p)) {
//            return false;
//        }
        __m256i pcmp = _mm256_cmpeq_epi32(a_p, b_p);
        unsigned bitmask = _mm256_movemask_epi8(pcmp);
        if (bitmask != 0xFFFFFFFFU) {
            return false;
        }
    }
    if (remainer) {
        for (PADO::VertexID v_i = v_i_bound; v_i < list_size; ++v_i) {
            if (a_list[v_i] != b_list[v_i]) {
                return false;
            }
        }
    }

    return true;

#else

// Sequential version
    PADO::VertexID v_i_bound = a_list.size();
    for (PADO::VertexID v_i = 0; v_i < v_i_bound; ++v_i) {
        if (a_list[v_i] != b_list[v_i]) {
            return false;
        }
    }

    return true;
#endif
}

void eliminate_vertices(char *argv[])
//void eliminate_vertices(char *input_filename, char *output_filename)
{
    double time_running = -PADO::WallTimer::get_time_mark();
    char *input_filename = argv[1];
    char *output_filename = argv[2];
    char *log_filename = argv[3];


    std::ifstream fin(input_filename);
    if (!fin.is_open()) {
        fprintf(stderr,
                "Error: cannot open file %s\n", input_filename);
        exit(EXIT_FAILURE);
    }

    std::ofstream fout(output_filename);
    if(!fout.is_open()) {
        fprintf(stderr,
                "Error: cannot create file %s\n", output_filename);
        exit(EXIT_FAILURE);
    }

    std::ifstream fv_in(log_filename);
    std::fstream fv_out(log_filename, std::ios::app);

    std::vector<uint8_t> is_redundant;
    {
        // Read the graph.
        // Get the num_v at first.
        PADO::VertexID num_v = 0;
        PADO::EdgeID num_e = 0;
        std::string line;
        while (std::getline(fin, line)) {
            if (line[0] == '#' || line[0] == '%' || line[0] == '+' || line[0] == '-') {
                continue;
            }
            std::istringstream iss(line);
            PADO::VertexID head;
            PADO::VertexID tail;
            iss >> head >> tail;
            num_v = std::max(num_v, std::max(head, tail) + 1);
            ++num_e;
        }
        printf("intput: num_v: ");
        std::cout << num_v;
        printf(" num_e: ");
        std::cout << num_e << std::endl;

        // Read the graph again, get the adjacency_list.
        fin.clear();
        fin.seekg(0);
        std::vector< std::vector<PADO::VertexID> > adjacency_list(num_v);
//        std::vector< std::set<PADO::VertexID> > adjacency_list(num_v);
        for (PADO::VertexID v_id = 0; v_id < num_v; ++v_id) {
            adjacency_list[v_id].push_back(v_id);
//            adjacency_list[v_id].insert(v_id);
        }
        while (std::getline(fin, line)) {
            if (line[0] == '#' || line[0] == '%' || line[0] == '+' || line[0] == '-') {
                continue;
            }
            std::istringstream iss(line);
            PADO::VertexID head;
            PADO::VertexID tail;
            iss >> head >> tail;
            adjacency_list[head].push_back(tail);
            adjacency_list[tail].push_back(head);
//            adjacency_list[head].insert(tail);
//            adjacency_list[tail].insert(head);
        }
        {// sort out edges
#pragma omp parallel for
            for (PADO::VertexID v_id = 0; v_id < num_v; ++v_id) {
                std::sort(adjacency_list[v_id].begin(), adjacency_list[v_id].end());
            }
        }

        double time_check = -PADO::WallTimer::get_time_mark();
        // Check every vertex if it is redundant.
        is_redundant.resize(num_v, 0);
        PADO::VertexID start_v = 0;
        {// Read the previous results
            /**
             * The format of the redundant_vertices file
             * <v_root> <size_r> <v0> <v1> ... (totally size_r vertices, size_r could be 0)
             * ...
             */
            PADO::VertexID v_root = 0;
            PADO::VertexID size_r = 0;
            PADO::VertexID v_dump = 0;
            while (fv_in >> v_root >> size_r) {
                if (v_root > start_v) {
                    start_v = v_root;
                }
                if (0 == size_r) {
                    continue;
                }
                for (PADO::VertexID v_i = 0; v_i < size_r; ++v_i) {
                    fv_in >> v_dump;
                    is_redundant[v_dump] = 1;
                }
            }
        }
        std::vector<PADO::VertexID> redundants_queue(num_v);
        PADO::VertexID end_redundants_queue = 0;

        for (PADO::VertexID a_v = start_v; a_v < num_v; ++a_v) {
//        for (PADO::VertexID a_v = 0; a_v < num_v; ++a_v) {
            if (is_redundant[a_v]) {
                continue;
            }
            const auto &a_set = adjacency_list[a_v];
#pragma omp parallel for
            for (PADO::VertexID b_v = a_v + 1; b_v < num_v; ++b_v) {
                if (is_redundant[b_v]) {
                    continue;
                }
                const auto &b_set = adjacency_list[b_v];
                if (a_set.size() != b_set.size()) {
                    continue;
                }
                if (are_equivalent(a_set, b_set)) {
//                if (a_set == b_set) {
                    is_redundant[b_v] = 1;
                    PADO::TS_enqueue(
                            redundants_queue,
                            end_redundants_queue,
                            b_v);
                }
            }

            // Save to the log file.
            fv_out << std::endl
                    << a_v << " " << end_redundants_queue;
            if (!end_redundants_queue) {
                continue;
            }
            for (PADO::VertexID v_i = 0; v_i < end_redundants_queue; ++v_i) {
                fv_out << " " << redundants_queue[v_i];
            }
            end_redundants_queue = 0;
        }
        time_check += PADO::WallTimer::get_time_mark();
        printf("time_check(s.): %f\n", time_check);
    }

    printf("Writing...\n");
    {
        // Read the graph again, do the reduction.
        std::vector< std::pair<PADO::VertexID, PADO::VertexID> > edge_list;
        PADO::VertexID num_v = 0;
        PADO::EdgeID num_e = 0;
        fin.clear();
        fin.seekg(0);
        std::string line;
        while (std::getline(fin, line)) {
            if (line[0] == '#' || line[0] == '%' || line[0] == '+' || line[0] == '-') {
                continue;
            }
            std::istringstream iss(line);
            PADO::VertexID head;
            PADO::VertexID tail;
            iss >> head >> tail;
            if (is_redundant[head] || is_redundant[tail]) {
                continue;
            }
            num_v = std::max(num_v, std::max(head, tail) + 1);
            ++num_e;
            edge_list.emplace_back(head, tail);
        }
        std::cout << "output: num_v: " << num_v << " num_e: " << num_e << std::endl;

        // Write into the binary file.
        fout.write(reinterpret_cast<char *>(&num_v), sizeof(num_v));
        fout.write(reinterpret_cast<char *>(&num_e), sizeof(num_e));
        for (const auto &edge : edge_list) {
            PADO::VertexID head = edge.first;
            PADO::VertexID tail = edge.second;
            fout.write(reinterpret_cast<char *>(&head), sizeof(head));
            fout.write(reinterpret_cast<char *>(&tail), sizeof(tail));
        }
    }

    time_running += PADO::WallTimer::get_time_mark();
    printf("running_time(s.): %f\n", time_running);
}

int main(int argc, char *argv[])
{
    if (argc < 4) {
        fprintf(stderr,
                "Usage: %s <input_edgelist> <output_binary> <log_file>\n"
                "\t<input_edgelist> should be unweighted.\n",
                argv[0]);
        exit(EXIT_FAILURE);
    }

//    {
//        int max_threads = omp_get_max_threads();
//        printf("max_threads: %d\n", max_threads);
//    }
    setbuf(stdout, nullptr); // stdout no buffer
    omp_set_num_threads(omp_get_max_threads());
    eliminate_vertices(argv);

    return EXIT_SUCCESS;
}