#CXX = g++
#CXXFLAGS =	-O0 -g -Wall -Wextra -fmessage-length=0 -std=c++14 -fopenmp

CXX = icpc
CXXFLAGS =	-O3 -g -w2 -fmessage-length=0 -std=c++14 -fopenmp

OBJS =		pado.o

#LIBS = /usr/local/lib/libpapi.a

INCLUDES_DIR = includes
INCLUDES = -I$(INCLUDES_DIR)

TARGET =	pado query_distance

.PHONY: all clean

all:	$(TARGET) 

pado: $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LIBS)
	
pado.o: pado.cpp $(INCLUDES_DIR)/*.h
	$(CXX) $(CXXFLAGS) -o $@ -c $< $(INCLUDES)	

query_distance: query_distance.cpp $(INCLUDES_DIR)/*.h
	$(CXX) $(CXXFLAGS) -o $@ $< $(INCLUDES)

clean:
	rm -f $(OBJS) $(TARGET)

# -qopt-report -qopt-report-phase=vec
