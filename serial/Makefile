CC = g++

CFLAGS = -g -std=c++20 -I. \
		 -fopenmp \
		 -Wall -Wextra \
		 -Wcast-align -Wconversion -Wfloat-equal -Wformat=2 -Wnull-dereference -Wshadow -Wsign-conversion -Wswitch-enum -Wundef -Wunreachable-code -Wunused \
		 -pthread
# -fsanitize=address -fsanitize=undefined 

all: life3d

life3d: main.cpp
	$(CC) $(CFLAGS) $(SLEEP) -o $@ $^

%.o: %.cpp %.hpp
	$(CC) $(CFLAGS) -c ${@:.o=.cpp} -o $@

clean:
	rm -f *.o life3d

format:
	clang-format -i *.cpp *.hpp
