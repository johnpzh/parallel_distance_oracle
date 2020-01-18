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
#include <set>
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
void eliminate_vertices(char *input_filename, char *output_filename)
{
    double time_running = -PADO::WallTimer::get_time_mark();

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

    std::vector<uint8_t> is_redundant;
//    std::vector<bool> is_redundant;
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
        std::vector< std::set<PADO::VertexID> > adjacency_list(num_v);
        for (PADO::VertexID v_id = 0; v_id < num_v; ++v_id) {
            adjacency_list[v_id].insert(v_id);
        }
        while (std::getline(fin, line)) {
            if (line[0] == '#' || line[0] == '%' || line[0] == '+' || line[0] == '-') {
                continue;
            }
            std::istringstream iss(line);
            PADO::VertexID head;
            PADO::VertexID tail;
            iss >> head >> tail;
            adjacency_list[head].insert(tail);
            adjacency_list[tail].insert(head);
        }

        // Check every vertex if it is redundant.
        is_redundant.resize(num_v, 0);
//        is_redundant.resize(num_v, false);
//        // Coarse-grained check
//        // Prepare offsets
//        int num_threads = omp_get_max_threads();
//        PADO::VertexID batch_size = (num_v + num_threads - 1) / num_threads;
//        std::vector< PADO::VertexID > start_locations(num_threads);
//#pragma omp parallel for
//        for (int i_t = 0; i_t < num_threads; ++i_t) {
//            start_locations[i_t] = i_t * batch_size;
//        }
//        // Multiple-thread check
////#pragma omp parallel for
////        for (int i_t = 0; i_t < num_threads; ++i_t) {
//#pragma omp parallel
//        {
//            int i_t = omp_get_thread_num();
//            PADO::VertexID v_i_start = start_locations[i_t];
//            PADO::VertexID v_i_bound = i_t != num_threads - 1 ?
//                                        start_locations[i_t + 1] :
//                                        num_v;
//            for (PADO::VertexID v_i = v_i_start; v_i < v_i_bound; ++v_i) {
//                if (is_redundant[v_i]) {
//                    continue;
//                }
//                const auto &a_set = adjacency_list[v_i];
//                for (PADO::VertexID b_v = v_i + 1; b_v < v_i_bound; ++b_v) {
//                    if (is_redundant[b_v]) {
//                        continue;
//                    }
//                    const auto &b_set = adjacency_list[b_v];
//                    if (a_set.size() != b_set.size()) {
//                        continue;
//                    }
//                    if (a_set == b_set) {
//                        is_redundant[b_v] = true;
//                    }
//                }
//            }
//        }

//        {
//            if (adjacency_list[317039] == adjacency_list[317040]) {
////            if (adjacency_list[316770] == adjacency_list[316775]) {
////            if (adjacency_list[316849] == adjacency_list[316851]) {
//                printf("eq.\n");
//            } else {
//                printf("not eq.\n");
//            }
//            exit(1);
//        }
//        std::vector<PADO::VertexID> roots(num_v);
        // Coarse-grained check
        for (PADO::VertexID a_v = 0; a_v < num_v; ++a_v) {
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
                if (a_set == b_set) {
                    is_redundant[b_v] = 1;
//                    is_redundant[b_v] = true;
//                    roots[b_v] = a_v;
                }
            }

//            {
//                for (PADO::VertexID b_v = a_v + 1; b_v < num_v; ++b_v) {
//                    if (tmp_is_redundant[b_v]) {
//                        continue;
//                    }
//                    const auto &b_set = adjacency_list[b_v];
//                    if (a_set.size() != b_set.size()) {
//                        continue;
//                    }
//                    if (a_set == b_set) {
//                        tmp_is_redundant[b_v] = true;
//                    }
//                }
//
//                for (PADO::VertexID b_v = a_v + 1; b_v < num_v; ++b_v) {
//                    if (tmp_is_redundant[b_v] != is_redundant[b_v]) {
//                        std::cout << "a_v: " << a_v <<
//                                    " b_v: " << b_v <<
//                                    " tmp_is_redundant: " << tmp_is_redundant[b_v] <<
//                                    " is_redundant: " << is_redundant[b_v] << std::endl;
//                        std::cout << "a_set ?= b_set: " << (bool) (adjacency_list[a_v] == adjacency_list[b_v]) << std::endl;
//                        exit(1);
//                    }
//                }
//            }
        }

//        // Fine-grained check
//        for (PADO::VertexID a_v = 0; a_v < num_v; ++a_v) {
//            if (is_redundant[a_v]) {
//                continue;
//            }
//            const auto &a_set = adjacency_list[a_v];
//            for (PADO::VertexID b_v = a_v + 1; b_v < num_v; ++b_v) {
//                if (is_redundant[b_v]) {
//                    continue;
//                }
//                const auto &b_set = adjacency_list[b_v];
//                if (a_set.size() != b_set.size()) {
//                    continue;
//                }
//                if (a_set == b_set) {
//                    is_redundant[b_v] = true;
//                }
//            }
//        }
//        {
//            PADO::VertexID count = 0;
//            for (PADO::VertexID v_i = 0; v_i < num_v; ++v_i) {
//                if (is_redundant[v_i]) {
//                    ++count;
//                    std::cout << "root: " << roots[v_i] << " v: " << v_i << std::endl;
//                }
//            }
//            printf("count_reduncant: %u\n", count);
//            exit(1);
//        }
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
    if (argc < 3) {
        fprintf(stderr,
                "Usage: %s <input_edgelist> <output_binary>\n"
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
    eliminate_vertices(argv[1], argv[2]);

    return EXIT_SUCCESS;
}