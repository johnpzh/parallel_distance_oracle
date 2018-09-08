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
#include "globals.h"

using std::vector;

namespace PADO {

class IndexType {
private:
	//	idi *vertices = nullptr;
	//	weighti *distances = nullptr;
	vector<idi> vertices;
	vector<weighti> distances;
	idi size = 0;

//	void construct(idi size);
public:
	IndexType() = default;
//	explicit IndexType(idi size);

	void add_label_seq(idi v, idi d)
	{
		++size;
		vertices.push_back(v);
		distances.push_back(d);
	}
	void add_label_par(idi v, idi d)
	{
//		idi
		//TODO:
	}

	idi get_label_ith_v(idi i) const
	{
		return vertices[i];
	}

	weighti get_label_ith_d(idi i) const
	{
		return distances[i];
	}

	weighti get_last_label_d() const
	{
		return distances[size - 1];
	}

	idi get_size() const
	{
		return size;
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
	for (idi i = 0; i < size; ++i) {
		printf("(%llu, %d)\n", vertices[i], distances[i]);
	}
}

// Not really use this class Label
class Label {
private:
	vector<IndexType> index;

public:
	Label() = default;
	Label(idi n);

	void construct(idi n);
	void ith_add_label_seq(idi i, idi v, idi d)
	{
		IndexType &index_i = index[i];
		index_i.add_label_seq(v, d);
	}
	void ith_add_label_par(idi i, idi v, idi d);

	idi ith_get_last_label_d(idi i)
	{
		return index[i].get_last_label_d();
	}
}; // class Label

Label::Label(idi n)
{
	construct(n);
}
void Label::construct(idi n)
{
	index.resize(n);
}
}; // namespace PADO


#endif /* INCLUDES_INDEX_H_ */
