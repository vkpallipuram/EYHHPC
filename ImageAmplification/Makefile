# The variable CC specifies which compiler will be used.
# # (because different unix systems may use different compilers)
 CC=g++
 CXX=nvcc
#
# # The variable CFLAGS specifies compiler options
# #   -c :    Only compile (don't link)
# #   -Wall:  Enable all warnings about lazy / dangerous C programming 
# #   -std=c99: Using newer C99 version of C programming language
CFLAGS=-lm -c -Wall -std=c99 -O3 -g
CXXFLAGS= -lm -O3 -use_fast_math -Xptxas -v -arch=sm_35

#
# # All of the .h header files to use as dependencies
# HEADERS=main.h image_template.h amplify.h gpu.h
#
# # All of the object files to produce as intermediary work
OBJECTS=main.o gpu.o
#
# # The final program to build
EXECUTABLE=amplify
#
# # --------------------------------------------
#
all: $(EXECUTABLE)
#
$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(EXECUTABLE)

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $<
%.o: %.cu $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $@ $<
clean:
	rm -rf *.o output*pgm *.c~ *.h~ $(EXECUTABLE)

