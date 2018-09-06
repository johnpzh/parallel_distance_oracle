//============================================================================
// Name        : pado.cpp
// Author      : Zhen Peng
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C, Ansi-style
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include "globals.h"
#include "graph.h"
#include "pado.h"

using namespace PADO;

void test()
{
	freopen("output.txt", "w", stdout); // test
	const char filename[] = "input.txt";
	Graph G(filename);
//	G.print();
	vector<idi> rank = {
			14,
			10,
			7,
			8,
			15,
			16,
			4,
			5,
			6,
			11,
			17,
			12,
			1,
			2,
			9,
			3,
			18,
			13,
			19,
			20
	};
	VertexCentricPLL(G, rank);
}

int main(void) {
	puts("Hello World!!!");
	test();
	puts("Done!");
	return EXIT_SUCCESS;
}
