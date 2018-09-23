import random
import sys
def create_graph(num_v):
	with open("output.txt", "w") as fout:
		random.seed();
		for v in range(num_v):
			for i in range(2):
				# fout.write(str(v) + " " + str(random.randint(0, num_v - 1)) + "\n")
				fout.write(str(v) + " " + str(v + i + 1) + "\n")


def adj_matrix_to_edgelist(filename):
	with open(filename, "r") as fin, open("output_edgelist.txt", "w") as fout:
		i = 0
		for line in fin:
			nums = line.split(', ')
			# print(nums)
			for j in range(i, len(nums)):
				if '1' == nums[j]:
					fout.write(str(i) + " " + str(j) + "\n")
			i += 1

def query_input():
	with open("output.query.edgelist.txt", "w") as fout:
		low = 0
		up = 69
		count = 200
		for i in range(count):
			fout.write("{} {}\n".format(random.randint(low, up), random.randint(low, up)))

if __name__ == '__main__':
	# if len(sys.argv) < 2:
	# 	print("Usage: python3 graph.py <arg>")
	# 	exit()
	# create_graph(int(sys.argv[1]))
	# adj_matrix_to_edgelist(sys.argv[1])
	query_input()
