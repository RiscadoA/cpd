CC = mpicc

CFLAGS = -std=c17 -I. \
		 -Wall -Wextra \
		 -Wcast-align -Wconversion -Wfloat-equal -Wformat=2 -Wnull-dereference -Wshadow -Wsign-conversion -Wswitch-enum -Wundef -Wunreachable-code -Wunused

ifeq ($(DEBUG), 1)
    CFLAGS += -g -O0 -fsanitize=address -fsanitize=undefined -DDEBUG
else
	CFLAGS += -O2
endif

all: life3d-mpi

life3d-mpi: life3d-mpi.c
	$(CC) $(CFLAGS) $(SLEEP) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c ${@:.o=.c} -o $@

clean:
	rm -f *.o life3d-mpi err_*.txt

format:
	clang-format -i *.c
