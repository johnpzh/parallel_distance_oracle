/*
 * graph.h
 *
 *  Created on: Sep 4, 2018
 *      Author: Zhen Peng
 */

#ifndef INCLUDES_GRAPH_H_
#define INCLUDES_GRAPH_H_

#include <vector>
#include <utility>
#include <string>
#include <fstream>
#include <sstream>
#include "globals.h"

using std::vector;
using std::string;
using std::getline;
using std::ifstream;
using std::istringstream;
using std::make_pair;
using std::pair;
using std::max;

namespace PADO
{
class Graph final {
private:
	idi num_v = 0;
	idi num_e = 0;
	idi *vertices = nullptr;
	idi *out_edges = nullptr;
	idi *out_degrees = nullptr;

	void construct(const char *filename);
	void construct(const vector< pair<idi, idi> > &edgeList);

public:
	// Constructor
	Graph() = default;
	explicit Graph(const char *filename);
	~Graph()
	{
		free(vertices);
		free(out_edges);
		free(out_degrees);
	}
	idi get_num_v() const
	{
		return num_v;
	}
	idi get_num_e() const
	{
		return num_e;
	}
	idi ith_get_edge(idi i, idi e) const
	{
		return out_edges[vertices[i] + e];
	}
	idi ith_get_out_degree(idi i) const
	{
		return out_degrees[i];
	}
	void print();
}; // class Graph

Graph::Graph(const char *filename)
{
	construct(filename);
}

// construcgt the graph from the edge list file
void Graph::construct(const char *filename)
{
	ifstream ifin(filename);
	if (ifin.bad()) {
		fprintf(stderr, "Error: cannot open file %s.\n", filename);
		exit(EXIT_FAILURE);
	}
	string line;
	idi head;
	idi tail;
	vector < pair<idi, idi> > edgeList;
	while (getline(ifin, line)) {
		if (line[0] == '#') {
			continue;
		}
		istringstream lin(line);
		lin >> head >> tail;
		edgeList.push_back(make_pair(head, tail));
	}
	construct(edgeList);
	edgeList.clear();
}
// construct the graph from edgeList
void Graph::construct(const vector< pair<idi, idi> > &edgeList)
{
	num_e = 2 * edgeList.size(); // Undirected Graph
	for (const auto &edge: edgeList) {
		num_v = max(num_v, max(edge.first, edge.second) + 1);
	}
	vertices = (idi *) malloc(num_v * sizeof(idi));
	out_edges = (idi *) malloc(num_e * sizeof(idi));
	out_degrees = (idi *) malloc(num_v * sizeof(idi));

	// Sort edgeList according to heads
	vector< vector<idi> > edge_tmp(num_v);
	for (const auto &edge: edgeList) {
		edge_tmp[edge.first].push_back(edge.second);
		edge_tmp[edge.second].push_back(edge.first);
	}
	// Get vertices and outEdges
	idi loc = 0;
	for (idi head = 0; head < num_v; ++head) {
		vertices[head] = loc;
		idi degree = edge_tmp[head].size();
		out_degrees[head] = degree;
		for (idi ei = 0; ei < degree; ++ei) {
			idi tail = edge_tmp[head][ei];
			out_edges[loc + ei] = tail;
		}
		loc += degree;
	}
	edge_tmp.clear();
}

// print every edge of the graph
void Graph::print()
{
	printf("num_v: %lld, num_e: %lld\n", num_v, num_e / 2);
	for (idi head = 0; head < num_v; ++head) {
		idi start_e = vertices[head];
		idi bound_e = start_e + out_degrees[head];
		for (idi e = start_e; e < bound_e; ++e) {
			idi tail = out_edges[e];
			printf("%llu %llu\n", head, tail);
		}
	}
}

} // namespace PADO



#endif /* INCLUDES_GRAPH_H_ */
