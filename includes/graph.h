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
#include <algorithm>
#include "globals.h"

using std::vector;
using std::string;
using std::getline;
using std::ifstream;
using std::istringstream;
using std::make_pair;
using std::pair;
using std::sort;
using std::max;

namespace PADO
{
class Graph final {
//private:
public:
	idi num_v = 0;
	idi num_e = 0;
	idi *vertices = nullptr;
	idi *out_edges = nullptr;
	idi *out_degrees = nullptr;

//	void construct(const char *filename);
	void construct(const vector< pair<idi, idi> > &edge_list);

//public:
	// Constructor
	Graph() = default;
	explicit Graph(const char *filename);
	explicit Graph(vector< pair<idi, idi> > &edge_list);
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

	// Rank according to degrees (right now)
	vector<idi> make_rank() const;
	// Remap vertex id according to its rank, 1 is the highest rank;
	vector<idi> id_transfer(const vector<idi> &rank);
	void print();
	
	// Test for Prof. Jin: how many vertices have degree 1 or 2 (and the proportion)
	void count_low_degrees()
	{
		idi count = 0;
		for (idi v_i = 0; v_i < num_v; ++v_i) {
//			if (1 == out_degrees[v_i]) {
//				++count;
//			}
			if (1 == out_degrees[v_i] || 2 == out_degrees[v_i]) {
				++count;
			}
		}
		printf("Num_of_V: %u Degree_1_or_2: %u %f%%\n\n", num_v, count, count * 100.0 / num_v);
		exit(EXIT_FAILURE);
	}
}; // class Graph

// construcgt the graph from the edge list file
Graph::Graph(const char *filename)
{
//	construct(filename);

	ifstream ifin(filename);
	if (!ifin.is_open()) {
		fprintf(stderr, "Error: cannot open file %s\n", filename);
		exit(EXIT_FAILURE);
	}
	string line;
	idi head;
	idi tail;
	vector < pair<idi, idi> > edge_list;
	while (getline(ifin, line)) {
		if (line[0] == '#' || line[0] == '%') {
			continue;
		}
		istringstream lin(line);
		lin >> head >> tail;
		edge_list.push_back(make_pair(head, tail));
	}
	construct(edge_list);
}

// construct the graph from edge_list
Graph::Graph(vector< pair<idi, idi> > &edge_list)
{
	construct(edge_list);
}

void Graph::construct(const vector< pair<idi, idi> > &edge_list)
{
	num_e = 2 * edge_list.size(); // Undirected Graph
//	num_e = edge_list.size(); // Directed Graph
	for (const auto &edge: edge_list) {
		num_v = max(num_v, max(edge.first, edge.second) + 1);
	}
	vertices = (idi *) malloc(num_v * sizeof(idi));
	out_edges = (idi *) malloc(num_e * sizeof(idi));
	out_degrees = (idi *) malloc(num_v * sizeof(idi));

	// Sort edge_list according to heads
	vector< vector<idi> > edge_tmp(num_v);
	for (const auto &edge: edge_list) {
		edge_tmp[edge.first].push_back(edge.second);
		edge_tmp[edge.second].push_back(edge.first); // Undirected Graph
	}
	// Get vertices and outEdges
	idi loc = 0;
	for (idi head = 0; head < num_v; ++head) {
		vertices[head] = loc;
		idi degree = edge_tmp[head].size();
		out_degrees[head] = degree;
		for (idi ei = 0; ei < degree; ++ei) {
			out_edges[loc + ei] = edge_tmp[head][ei];
		}
		loc += degree;
	}
}

// Rank according to degrees
vector<idi> Graph::make_rank() const
{
	vector< pair<double, idi> > degree2id;
	//vector< pair<idi, idi> > degree2id;
	for (idi v = 0; v < num_v; ++v) {
		// Add a random value here to diffuse nearby vertices, according to PLL's implementation.
		// Somehow it decreases the label size a little bit.
		degree2id.push_back(make_pair(out_degrees[v] + (double) rand() / RAND_MAX, v));
		//degree2id.push_back(make_pair(out_degrees[v], v));
	}
	sort(degree2id.rbegin(), degree2id.rend());
	vector<idi> rank(num_v);
	for (idi r = 0; r < num_v; ++r) {
		rank[degree2id[r].second] = r;
	}
	return rank;
}

vector<idi> Graph::id_transfer(const vector<idi> &rank)
{
	// The new edge list
	vector< vector<idi> > edge_list(num_v);
	for (idi v = 0; v < num_v; ++v) {
		idi new_v = rank[v];
		idi ei_start = vertices[v];
		idi ei_bound = ei_start + out_degrees[v];
		for (idi ei = ei_start; ei < ei_bound; ++ei) {
			idi new_w = rank[out_edges[ei]];
			edge_list[new_v].push_back(new_w);
		}
	}
	idi loc = 0;
	for (idi head = 0; head < num_v; ++head) {
		vertices[head] = loc;
		sort(edge_list[head].rbegin(), edge_list[head].rend()); // sort neighbors: lower rank first.
		idi degree = edge_list[head].size();
		out_degrees[head] = degree;
		for (idi ei = 0; ei < degree; ++ei) {
			out_edges[loc + ei] = edge_list[head][ei];
		}
		loc += degree;
	}

	vector<idi> rank2id(num_v);
	for (idi v = 0; v < num_v; ++v) {
		rank2id[rank[v]] = v;
	}
	return rank2id;
}

// print every edge of the graph
void Graph::print()
{
	printf("num_v: %u, num_e: %u\n", num_v, num_e / 2);
	for (idi head = 0; head < num_v; ++head) {
		idi start_e = vertices[head];
		idi bound_e = start_e + out_degrees[head];
		for (idi e = start_e; e < bound_e; ++e) {
			idi tail = out_edges[e];
			printf("%u %u\n", head, tail);
		}
	}
}
// End: class Graph

class WeightedGraph final {
//private:
public:
	idi num_v = 0;
	idi num_e = 0;
	idi *vertices = nullptr;
	idi *out_edges = nullptr;
	idi *out_degrees = nullptr;
	weighti *out_weights = nullptr;

	void construct(
			const vector< pair<idi, idi> > &edge_list, 
			const vector<weighti> &weight_list);

//public:
	// Constructor
	WeightedGraph() = default;
	explicit WeightedGraph(const char *filename);
	explicit WeightedGraph(
			const vector< pair<idi, idi> > &edge_list,
			const vector<weighti> &weight_list);
	~WeightedGraph()
	{
		free(vertices);
		free(out_edges);
		free(out_degrees);
		free(out_weights);
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

	// Rank according to degrees (right now)
	vector<idi> make_rank();
	// Remap vertex id according to its rank, 1 is the highest rank;
	vector<idi> id_transfer(const vector<idi> &rank);
	void print();
	
}; // class WeightedGraph

// construcgt the graph from the edge list file
WeightedGraph::WeightedGraph(const char *filename)
{
//	construct(filename);

	ifstream ifin(filename);
	if (!ifin.is_open()) {
		fprintf(stderr, "Error: cannot open file %s\n", filename);
		exit(EXIT_FAILURE);
	}
	string line;
	idi head;
	idi tail;
	inti weight;
	vector < pair<idi, idi> > edge_list;
	vector <weighti> weight_list;

	while (getline(ifin, line)) {
		if (line[0] == '#' || line[0] == '%') {
			continue;
		}
		istringstream lin(line);
		lin >> head >> tail >> weight;
		edge_list.push_back(make_pair(head, tail));
		weight_list.push_back(static_cast<weighti>(weight));
	}
	construct(edge_list, weight_list);
}

// construct the WeightedGraph from edge_list
WeightedGraph::WeightedGraph(
		const vector< pair<idi, idi> > &edge_list,
		const vector<weighti> &weight_list)
{
	construct(edge_list, weight_list);
}

void WeightedGraph::construct(
		const vector< pair<idi, idi> > &edge_list,
		const vector<weighti> &weight_list)
{
	num_e = 2 * edge_list.size(); // Undirected Graph
//	num_e = edge_list.size(); // Directed Graph
	for (const auto &edge: edge_list) {
		num_v = max(num_v, max(edge.first, edge.second) + 1);
	}
	vertices = (idi *) malloc(num_v * sizeof(idi));
	out_edges = (idi *) malloc(num_e * sizeof(idi));
	out_degrees = (idi *) malloc(num_v * sizeof(idi));
	out_weights = (weighti *) malloc(num_e * sizeof(weighti));

	// Sort edge_list according to heads
	vector< vector<idi> > edge_tmp(num_v);
	vector< vector<weighti> > weights_tmp(num_v);
	idi w_i = 0;
	for (const auto &edge: edge_list) {
		edge_tmp[edge.first].push_back(edge.second);
		edge_tmp[edge.second].push_back(edge.first); // Undirected Graph
		weights_tmp[edge.first].push_back(weight_list[w_i]);
		weights_tmp[edge.second].push_back(weight_list[w_i]); // Undirected Graph
		++w_i;
	}
	// Get vertices and outEdges
	idi loc = 0;
	for (idi head = 0; head < num_v; ++head) {
		vertices[head] = loc;
		idi degree = edge_tmp[head].size();
		out_degrees[head] = degree;
		for (idi ei = 0; ei < degree; ++ei) {
			out_edges[loc + ei] = edge_tmp[head][ei];
			out_weights[loc + ei] = weights_tmp[head][ei];
		}
		loc += degree;
	}
}

// Rank according to degrees
vector<idi> WeightedGraph::make_rank()
{
	vector< pair<double, idi> > degree2id;
	//vector< pair<idi, idi> > degree2id;
	for (idi v = 0; v < num_v; ++v) {
		// Add a random value here to diffuse nearby vertices, according to PLL's implementation.
		// Somehow it decreases the label size a little bit.
		degree2id.push_back(make_pair(out_degrees[v] + (double) rand() / RAND_MAX, v));
		//degree2id.push_back(make_pair(out_degrees[v], v));
	}
	sort(degree2id.rbegin(), degree2id.rend());
	vector<idi> rank(num_v);
	for (idi r = 0; r < num_v; ++r) {
		rank[degree2id[r].second] = r;
	}
	return rank;
}

// Function: transfer vertex IDs according to their ranks
vector<idi> WeightedGraph::id_transfer(const vector<idi> &rank)
{
	// The new edge list
//	vector< vector<idi> > edge_list(num_v);
	vector< vector< pair<idi, weighti> > > edge_list(num_v); // pair of (vertex id, edge weight)
	for (idi v = 0; v < num_v; ++v) {
		idi new_v = rank[v];
		idi ei_start = vertices[v];
		idi ei_bound = ei_start + out_degrees[v];
		for (idi ei = ei_start; ei < ei_bound; ++ei) {
			idi new_w = rank[out_edges[ei]];
//			edge_list[new_v].push_back(new_w);
			edge_list[new_v].push_back(make_pair(new_w, out_weights[ei]));
		}
	}
	idi loc = 0;
	for (idi head = 0; head < num_v; ++head) {
		vertices[head] = loc;
		sort(edge_list[head].rbegin(), edge_list[head].rend()); // sort neighbors: lower rank first.
		idi degree = edge_list[head].size();
		out_degrees[head] = degree;
		for (idi ei = 0; ei < degree; ++ei) {
			out_edges[loc + ei] = edge_list[head][ei].first;
//			out_edges[loc + ei] = edge_list[head][ei];
			out_weights[loc + ei] = edge_list[head][ei].second;
		}
		loc += degree;
	}

	vector<idi> rank2id(num_v);
	for (idi v = 0; v < num_v; ++v) {
		rank2id[rank[v]] = v;
	}
	return rank2id;
}

// print every edge of the graph
void WeightedGraph::print()
{
	printf("num_v: %u, num_e: %u\n", num_v, num_e / 2);
	for (idi head = 0; head < num_v; ++head) {
		idi start_e = vertices[head];
		idi bound_e = start_e + out_degrees[head];
		for (idi e = start_e; e < bound_e; ++e) {
			idi tail = out_edges[e];
			weighti wt = out_weights[e];
			printf("%u %u %u\n", head, tail, wt);
		}
	}
}
// End: class WeightedGraph

} // namespace PADO



#endif /* INCLUDES_GRAPH_H_ */
