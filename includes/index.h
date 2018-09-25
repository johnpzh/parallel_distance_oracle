/*
 * index.h
 *
 *  Created on: Sep 4, 2018
 *      Author: Zhen Peng
 */

#ifndef INCLUDES_INDEX_H_
#define INCLUDES_INDEX_H_

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <map>
#include "globals.h"

using std::vector;
using std::map;

namespace PADO {

//// IndexType using map
//class IndexType {
//private:
//
//	idi size = 0;
//	map<idi, weighti> labels;
//
//public:
//	IndexType() = default;
////	explicit IndexType(idi size);
//
//	void add_label_seq(idi v, idi d)
//	{
//		++size;
//		labels[v] = d;
//	}
//	void add_label_par(idi v, idi d)
//	{
//		//TODO:
//	}
//
//	weighti get_dist(idi v) const
//	{
////		return labels[v];
//		return labels.find(v)->second;
//	}
//	auto get_label_begin() const
//	{
//		return labels.begin();
//	}
//	auto get_label_end() const
//	{
//		return labels.end();
//	}
//
//	idi get_size() const
//	{
//		return size;
//	}
//
//	bool is_v_in_label(idi v) const
//	{
//		if (labels.end() == labels.find(v)) {
//			return false;
//		} else {
//			return true;
//		}
//	}
//
//	void print();
//
//}; // class IndexType
//
//void IndexType::print()
//{
//	for (const auto &l : labels) {
//		printf("(%llu, %d)\n", l.first, l.second);
//	}
//}

// IndexType using vector
class IndexType {
public:
//private:
	vector<idi> vertices;
	vector<weighti> distances;

//	vector<inti> batch_lens = vector<inti>(1, 0); // FIXME
//	idi size = 0;

//	void construct(idi size);
//public:
	IndexType() = default;
//	explicit IndexType(idi size);

	void add_label_seq(idi v, weighti d)
	{
//		++size;
		vertices.push_back(v);
		distances.push_back(d);
	}
//	void add_label_par(idi v, weighti d)
//	{
//		//TODO:
//	}

	idi get_label_ith_v(idi i) const
	{
		return vertices[i];
	}

	weighti get_label_ith_d(idi i) const
	{
		return distances[i];
	}

//	weighti get_last_label_d() const
//	{
//		return distances[size - 1];
//	}

	idi get_size() const
	{
//		return size;
		return vertices.size();
	}

	bool is_v_in_label(idi v) const
	{
		for (const idi &vt : vertices) {
			if (vt == v) {
				return true;
			}
		}
		return false;
	}

	void print();

}; // class IndexType

void IndexType::print()
{
	for (idi i = 0; i < vertices.size(); ++i) {
		printf("(%u, %d)\n", vertices[i], distances[i]);
	}
}

// End Class IndexType
}; // namespace PADO


#endif /* INCLUDES_INDEX_H_ */
