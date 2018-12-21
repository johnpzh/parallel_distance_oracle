CXX = g++
CXXFLAGS =	-O0 -g -Wall -Wextra -fmessage-length=0 -std=c++14 -fopenmp

#CXX = icpc
#CXXFLAGS =	-O3 -w2 -fmessage-length=0 -std=c++14 -fopenmp

OBJS =		pado.o

LIBS = /usr/local/lib/libpapi.a

INCLUDES_DIR = includes
INCLUDES = -I$(INCLUDES_DIR)

TARGET =	pado

.PHONY: all clean

all:	$(TARGET) 

$(TARGET):	$(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) $(LIBS)
	
pado.o: pado.cpp $(INCLUDES_DIR)/*.h
	$(CXX) $(CXXFLAGS) -o $@ -c $< $(INCLUDES)	

clean:
	rm -f $(OBJS) $(TARGET)
