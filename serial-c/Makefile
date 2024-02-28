CC = gcc

CFLAGS = -std=c17 -I. \
		 -fopenmp \
		 -Wall -Wextra \
		 -Wcast-align -Wconversion -Wfloat-equal -Wformat=2 -Wnull-dereference -Wshadow -Wsign-conversion -Wswitch-enum -Wundef -Wunreachable-code -Wunused

ifeq ($(DEBUG), 1)
    CFLAGS += -g -O0 -fsanitize=address -fsanitize=undefined
else
	CFLAGS += -O3
endif

all: life3d

life3d: life3d.c
	$(CC) $(CFLAGS) $(SLEEP) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c ${@:.o=.c} -o $@

clean:
	rm -f *.o life3d

format:
	clang-format -i *.c
