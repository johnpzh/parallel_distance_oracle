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
	with open("output.query.wikitalk.txt", "w") as fout:
		low = 0
		up = 2394385
		count = 1000
		for i in range(count):
			fout.write("{} {}\n".format(random.randint(low, up), random.randint(low, up)))

def add_weights_to_graph():
    if len(sys.argv) < 5:
        print("Usage: python3 graph.py <input_graph> <output_graph> <lower_weight> <upper_weight>")
        exit()
    with open(sys.argv[1], "r") as fin, \
        open(sys.argv[2], "w") as fout:
            lower_wt = int(sys.argv[3])
            upper_wt = int(sys.argv[4])
            random.seed()
            for in_line in fin:
                in_line = in_line.strip()
                if in_line[0] == '#' or in_line[0] == '%':
                    fout.write("{}\n".format(in_line))
                    continue
                wt = random.randint(lower_wt, upper_wt)
                out_line = in_line + " " + str(wt) + "\n"
                fout.write(out_line)

if __name__ == '__main__':
	# if len(sys.argv) < 2:
	# 	print("Usage: python3 graph.py <arg>")
	# 	exit()
	# create_graph(int(sys.argv[1]))
	# adj_matrix_to_edgelist(sys.argv[1])
	# query_input()
    add_weights_to_graph()
