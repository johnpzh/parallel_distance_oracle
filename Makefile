CXX = /usr/local/bin/g++-8

CXXFLAGS =	-O3 -g -Wall -Wextra -fmessage-length=0 -std=c++14

OBJS =		pado.o

LIBS =

INCLUDES_DIR = includes
INCLUDES = -I$(INCLUDES_DIR)

TARGET =	pado

.PHONY: all clean

all:	$(TARGET) 

$(TARGET):	$(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LIBS)
	
pado.o: pado.cpp $(INCLUDES_DIR)/*.h
	$(CXX) $(CXXFLAGS) -o $@ -c $< $(INCLUDES)	

clean:
	rm -f $(OBJS) $(TARGET)
