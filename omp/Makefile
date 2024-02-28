CC = g++

CFLAGS = -std=c++20 -I. \
		 -fopenmp \
		 -Wall -Wextra \
		 -Wcast-align -Wconversion -Wfloat-equal -Wformat=2 -Wnull-dereference -Wshadow -Wsign-conversion -Wswitch-enum -Wundef -Wunreachable-code -Wunused \
		 -pthread

ifeq ($(DEBUG), 1)
    CFLAGS += -g -O0 -fsanitize=address -fsanitize=undefined
else
	CFLAGS += -O2
endif

all: life3d-omp

life3d-omp: life3d-omp.cpp
	$(CC) $(CFLAGS) $(SLEEP) -o $@ $^

%.o: %.cpp
	$(CC) $(CFLAGS) -c ${@:.o=.cpp} -o $@

clean:
	rm -f *.o life3d-omp

format:
	clang-format -i *.cpp
