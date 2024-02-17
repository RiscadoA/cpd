CC = g++

CFLAGS = -std=c++20 -I. -O3 \
		 -fopenmp \
		 -Wall -Wextra \
		 -Wcast-align -Wconversion -Wfloat-equal -Wformat=2 -Wnull-dereference -Wshadow -Wsign-conversion -Wswitch-enum -Wundef -Wunreachable-code -Wunused \
		 -pthread

all: life3d

life3d: life3d.cpp
	$(CC) $(CFLAGS) $(SLEEP) -o $@ $^

%.o: %.cpp %.hpp
	$(CC) $(CFLAGS) -c ${@:.o=.cpp} -o $@

clean:
	rm -f *.o life3d

format:
	clang-format -i *.cpp *.hpp
