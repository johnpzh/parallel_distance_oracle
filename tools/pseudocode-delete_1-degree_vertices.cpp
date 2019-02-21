/*
   A utility program to delete 1-degree vertices recursively from a graph, then 
   store in to a new file.
   02/16/2019
*/
#include <vector>
#include <utility>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>

using std::vector;
using std::string;
using std::getline;
using std::ifstream;
using std::ofstream;
using std::istringstream;
using std::make_pair;
using std::pair;
using std::sort;
using std::max;

typedef uint32_t idi;
const idi INF = (idi) -1;

class Graph final {
private:
	idi num_v = 0;
	idi num_e = 0;
	idi *vertices = nullptr;
	idi *out_edges = nullptr;
	idi *out_degrees = nullptr;

//	void construct(const char *filename);
	void construct(const vector< pair<idi, idi> > &edge_list);

public:
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
	void delete_1_degree_vertices(
			const char *fin_name,
			const char *fout_name);
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

void Graph::delete_1_degree_vertices(
		const char *fin_name,
		const char *fout_name)
{
	vector<bool> is_deleted(num_v, false);
	for (idi v_id = 0; v_id < num_v; ++v_id) {
		idi next_id = INF;
		if (!is_deleted[v_id] && out_degrees[v_id] == 1) {
			// If v_id's degree == 1
			next_id = v_id;
		}

		// DFS from v_id
		while (next_id != INF) {
			idi v_id = next_id;
			is_deleted[v_id] = true;
			idi w_id = INF;
			// Traverse v_id's all neighbors.
			// Find one of v_id's neighbors which is not deleted yet.
			idi start_e_i = vertices[v_id];
			idi bound_e_i = start_e_i + out_degrees[v_id];
			for (idi e_i = start_e_i; e_i < bound_e_i; ++e_i) {
				w_id = out_edges[e_i];
				if (!is_deleted[w_id]) {
					break;
				}
			} // then w_id is v_id's neighbor which is yet deleted.
			if (w_id != INF && out_degrees[w_id] == 2) {
				// w_id should be an 1-degree vertex after v_id is deleted.
				next_id = w_id;
			} else {
				// w_id is not the next 1-degree vertex.
				next_id = INF;
			}
		}
	} // Then the is_deleted flag array is done.

	// Read the original file, then write to a new file
	ifstream fin(fin_name);
	if (!fin.is_open()) {
		fprintf(stderr, "Error: cannot open file %s\n", fin_name);
		exit(EXIT_FAILURE);
	}
	ofstream fout(fout_name);
	if (!fout.is_open()) {
		fprintf(stderr, "Error: cannot open file %s\n", fout_name);
		exit(EXIT_FAILURE);
	}

	string line;
	idi head;
	idi tail;
	vector< vector<idi> > matrix(num_v);
	while (getline(fin, line)) {
		if (line[0] == '#' || line[0] == '%') {
			continue;
		}
		istringstream lin(line);
		lin >> head >> tail;
		if (is_deleted[head] || is_deleted[tail]) {
			continue;
		}
		fout << head << ' ' << tail << std::endl;
		matrix[head].push_back(tail);
		matrix[tail].push_back(head);
	}

	// Count the 2-degree vertices and deleted vertices
	idi deleted = 0;
	idi two_degrees = 0;
	for (idi v_id = 0; v_id < num_v; ++v_id) {
		if (matrix[v_id].size() == 0) {
			++deleted;
		} else if (matrix[v_id].size() == 2) {
			++two_degrees;
		}
	}
	setlocale(LC_NUMERIC, "");
	printf("deleted: %'u %.2f%% two_degrees: %'u %.2f%%\n", deleted, 100.0 * deleted / num_v, two_degrees, 100.0 * two_degrees / (num_v - deleted));

	fout.close();
	fin.close();
}

// End: class Graph

int main(int argc, char *argv[])
{
	if (argc < 3) {
		fprintf(stderr, "Usage: ./delete_1-degree_vertices <input_file> <output_file>\n");
		exit(EXIT_FAILURE);
	}

	printf("Reading...\n");
	Graph G(argv[1]);
	printf("Deleting...\n");
	G.delete_1_degree_vertices(argv[1], argv[2]);

	return EXIT_SUCCESS;
}
