#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N_SPECIES 9

unsigned int internal_seed;

struct maximum {
  int generation;
  long long count;
};

/**
 * @brief Initialize the random number generator.
 * @param input_seed Seed for the random number generator.
 */
void init_random(int input_seed) { internal_seed = (unsigned int)(input_seed + 987654321); }

/**
 * @brief Generate a random number between 0 and 1.
 * @return Random number.
 */
float get_random() {
  int seed_in = (int)internal_seed;

  internal_seed ^= (internal_seed << 13);
  internal_seed ^= (internal_seed >> 17);
  internal_seed ^= (internal_seed << 5);

  return (float)(0.5 + 0.2328306e-09 * (double)(seed_in + (int)internal_seed));
}

/**
 * @brief Allocate memory a 3D GOL grid.
 * @param N Side length of the grid.
 */
unsigned char *alloc_grid(int N) {
  size_t side = (size_t)N + 2;
  return (unsigned char *)malloc(side * side * side * sizeof(unsigned char));
}

/**
 * @brief Free the memory of a 3D GOL grid.
 * @param grid Grid to be freed.
 */
void free_grid(unsigned char *grid) { free(grid); }

#define linear_from_3d(S, x, y, z) ((x) * (S) * (S) + (y) * (S) + (z))

#define read_neighbor(grid, S, c, dx, dy, dz) grid[(c) + linear_from_3d((S), (dx), (dy), (dz))]

#define is_neighbor_alive(grid, S, c, dx, dy, dz)                                                  \
  !!read_neighbor((grid), (S), (c), (dx), (dy), (dz))

#define fast_max(x, y) ((x) - (((x) - (y)) & ((x) - (y)) >> 31))

/**
 * @brief Randomize the grid with a given density.
 * @param grid Grid to be randomized.
 * @param N Side length of the grid.
 * @param density Density of the grid.
 * @param input_seed Seed for the random number generator.
 */
void randomize_grid(unsigned char *grid, int N, float density, int input_seed) {
  init_random(input_seed);
  for (int x = 1; x <= N; x++)
    for (int y = 1; y <= N; y++)
      for (int z = 1; z <= N; z++)
        if (get_random() < density)
          grid[linear_from_3d(N + 2, x, y, z)] = (unsigned char)(get_random() * N_SPECIES) + 1;
}

void print_grid(const char *title, unsigned char *grid, int N) {
  printf("%s\n---\n", title);
  for (int x = 0; x <= N + 1; ++x) {
    printf("x = %d\n", x);
    for (int y = 0; y <= N + 1; ++y) {
      for (int z = 0; z <= N + 1; ++z) {
        if (grid[linear_from_3d(N + 2, x, y, z)]) {
          printf("%d ", grid[linear_from_3d(N + 2, x, y, z)]);
        } else {
          printf("  ");
        }
      }
      printf("\n");
    }
    printf("\n");
  }
}

int main(int argc, char **argv) {
  if (argc != 5) {
    fprintf(stderr, "Usage: %s <generations> <side> <density> <seed>\n", argv[0]);
    return 1;
  }

  int generations = atoi(argv[1]);
  int N = atoi(argv[2]);
  float density = (float)atof(argv[3]);
  int seed = atoi(argv[4]);

  if (N <= 0 || density < 0 || density > 1) {
    fprintf(stderr, "Invalid arguments\n");
    return 1;
  }

  /* Initialize the grids */
  unsigned char *previous = alloc_grid(N);
  unsigned char *next = alloc_grid(N);
  randomize_grid(previous, N, density, seed);

  /* Start tracking time */
  double time = -omp_get_wtime();

  /* Count how many of each species is alive */
  long long total_count[N_SPECIES + 1] = {0};
  for (int x = 1; x <= N; ++x) {
    for (int y = 1; y <= N; ++y) {
      for (int z = 1; z <= N; ++z) {
        total_count[previous[linear_from_3d(N + 2, x, y, z)]] += 1;
      }
    }
  }

  /* Prepare the maximums array */
  struct maximum maximums[N_SPECIES + 1];
  for (int i = 1; i <= N_SPECIES; i++) {
    maximums[i].generation = 0;
    maximums[i].count = total_count[i];
    total_count[i] = 0;
  }

  /* Run the simulation */
  for (int g = 1; g <= generations; g++) {
    /* Stretch the grid in the x axis */
    for (int u = 1; u <= N; ++u) {
      for (int v = 1; v <= N; ++v) {
        previous[linear_from_3d(N + 2, 0, u, v)] = previous[linear_from_3d(N + 2, N, u, v)];
        previous[linear_from_3d(N + 2, N + 1, u, v)] = previous[linear_from_3d(N + 2, 1, u, v)];
      }
    }

    /* Stretch the grid in the y axis */
    for (int u = 0; u <= N + 1; ++u) {
      for (int v = 1; v <= N; ++v) {
        previous[linear_from_3d(N + 2, u, 0, v)] = previous[linear_from_3d(N + 2, u, N, v)];
        previous[linear_from_3d(N + 2, u, N + 1, v)] = previous[linear_from_3d(N + 2, u, 1, v)];
      }
    }

    /* Stretch the grid in the z axis */
    for (int u = 0; u <= N + 1; ++u) {
      for (int v = 0; v <= N + 1; ++v) {
        previous[linear_from_3d(N + 2, u, v, 0)] = previous[linear_from_3d(N + 2, u, v, N)];
        previous[linear_from_3d(N + 2, u, v, N + 1)] = previous[linear_from_3d(N + 2, u, v, 1)];
      }
    }

    /* Update the cells */
    for (int x = 1; x <= N; ++x) {
      for (int y = 1; y <= N; ++y) {
        for (int z = 1; z <= N; ++z) {
          int c = linear_from_3d(N + 2, x, y, z);
          unsigned char current = previous[c];

          if (current) {
            int live_neighbors = 0;
            live_neighbors += is_neighbor_alive(previous, N + 2, c, 0, 0, -1);
            live_neighbors += is_neighbor_alive(previous, N + 2, c, 0, 0, 1);
            live_neighbors += is_neighbor_alive(previous, N + 2, c, -1, -1, -1);
            live_neighbors += is_neighbor_alive(previous, N + 2, c, -1, -1, 0);
            live_neighbors += is_neighbor_alive(previous, N + 2, c, -1, -1, 1);
            live_neighbors += is_neighbor_alive(previous, N + 2, c, -1, 0, -1);
            live_neighbors += is_neighbor_alive(previous, N + 2, c, -1, 0, 0);
            live_neighbors += is_neighbor_alive(previous, N + 2, c, -1, 0, 1);
            live_neighbors += is_neighbor_alive(previous, N + 2, c, -1, 1, -1);
            live_neighbors += is_neighbor_alive(previous, N + 2, c, -1, 1, 0);
            live_neighbors += is_neighbor_alive(previous, N + 2, c, -1, 1, 1);
            live_neighbors += is_neighbor_alive(previous, N + 2, c, 0, -1, -1);
            live_neighbors += is_neighbor_alive(previous, N + 2, c, 0, -1, 0);
            live_neighbors += is_neighbor_alive(previous, N + 2, c, 0, -1, 1);
            live_neighbors += is_neighbor_alive(previous, N + 2, c, 0, 1, -1);
            live_neighbors += is_neighbor_alive(previous, N + 2, c, 0, 1, 0);
            live_neighbors += is_neighbor_alive(previous, N + 2, c, 0, 1, 1);
            live_neighbors += is_neighbor_alive(previous, N + 2, c, 1, -1, -1);
            live_neighbors += is_neighbor_alive(previous, N + 2, c, 1, -1, 0);
            live_neighbors += is_neighbor_alive(previous, N + 2, c, 1, -1, 1);
            live_neighbors += is_neighbor_alive(previous, N + 2, c, 1, 0, -1);
            live_neighbors += is_neighbor_alive(previous, N + 2, c, 1, 0, 0);
            live_neighbors += is_neighbor_alive(previous, N + 2, c, 1, 0, 1);
            live_neighbors += is_neighbor_alive(previous, N + 2, c, 1, 1, -1);
            live_neighbors += is_neighbor_alive(previous, N + 2, c, 1, 1, 0);
            live_neighbors += is_neighbor_alive(previous, N + 2, c, 1, 1, 1);
            if (live_neighbors <= 4 || live_neighbors > 13) {
              current = 0;
            }
          } else {
            int neighbor_count[N_SPECIES + 1] = {0};
            neighbor_count[read_neighbor(previous, N + 2, c, 0, 0, -1)] += 1;
            neighbor_count[read_neighbor(previous, N + 2, c, 0, 0, 1)] += 1;
            neighbor_count[read_neighbor(previous, N + 2, c, -1, -1, -1)] += 1;
            neighbor_count[read_neighbor(previous, N + 2, c, -1, -1, 0)] += 1;
            neighbor_count[read_neighbor(previous, N + 2, c, -1, -1, 1)] += 1;
            neighbor_count[read_neighbor(previous, N + 2, c, -1, 0, -1)] += 1;
            neighbor_count[read_neighbor(previous, N + 2, c, -1, 0, 0)] += 1;
            neighbor_count[read_neighbor(previous, N + 2, c, -1, 0, 1)] += 1;
            neighbor_count[read_neighbor(previous, N + 2, c, -1, 1, -1)] += 1;
            neighbor_count[read_neighbor(previous, N + 2, c, -1, 1, 0)] += 1;
            neighbor_count[read_neighbor(previous, N + 2, c, -1, 1, 1)] += 1;
            neighbor_count[read_neighbor(previous, N + 2, c, 0, -1, -1)] += 1;
            neighbor_count[read_neighbor(previous, N + 2, c, 0, -1, 0)] += 1;
            neighbor_count[read_neighbor(previous, N + 2, c, 0, -1, 1)] += 1;
            neighbor_count[read_neighbor(previous, N + 2, c, 0, 1, -1)] += 1;
            neighbor_count[read_neighbor(previous, N + 2, c, 0, 1, 0)] += 1;
            neighbor_count[read_neighbor(previous, N + 2, c, 0, 1, 1)] += 1;
            neighbor_count[read_neighbor(previous, N + 2, c, 1, -1, -1)] += 1;
            neighbor_count[read_neighbor(previous, N + 2, c, 1, -1, 0)] += 1;
            neighbor_count[read_neighbor(previous, N + 2, c, 1, -1, 1)] += 1;
            neighbor_count[read_neighbor(previous, N + 2, c, 1, 0, -1)] += 1;
            neighbor_count[read_neighbor(previous, N + 2, c, 1, 0, 0)] += 1;
            neighbor_count[read_neighbor(previous, N + 2, c, 1, 0, 1)] += 1;
            neighbor_count[read_neighbor(previous, N + 2, c, 1, 1, -1)] += 1;
            neighbor_count[read_neighbor(previous, N + 2, c, 1, 1, 0)] += 1;
            neighbor_count[read_neighbor(previous, N + 2, c, 1, 1, 1)] += 1;

            int live_neighbors = 26 - neighbor_count[0];
            if (live_neighbors >= 7 && live_neighbors <= 10) {
              neighbor_count[1] = (neighbor_count[1] << 4) | (N_SPECIES - 1);
              neighbor_count[2] = (neighbor_count[2] << 4) | (N_SPECIES - 2);
              neighbor_count[3] = (neighbor_count[3] << 4) | (N_SPECIES - 3);
              neighbor_count[4] = (neighbor_count[4] << 4) | (N_SPECIES - 4);
              neighbor_count[5] = (neighbor_count[5] << 4) | (N_SPECIES - 5);
              neighbor_count[6] = (neighbor_count[6] << 4) | (N_SPECIES - 6);
              neighbor_count[7] = (neighbor_count[7] << 4) | (N_SPECIES - 7);
              neighbor_count[8] = (neighbor_count[8] << 4) | (N_SPECIES - 8);
              neighbor_count[9] = (neighbor_count[9] << 4) | (N_SPECIES - 9);

              int maximum = neighbor_count[1];
              maximum = fast_max(maximum, neighbor_count[2]);
              maximum = fast_max(maximum, neighbor_count[3]);
              maximum = fast_max(maximum, neighbor_count[4]);
              maximum = fast_max(maximum, neighbor_count[5]);
              maximum = fast_max(maximum, neighbor_count[6]);
              maximum = fast_max(maximum, neighbor_count[7]);
              maximum = fast_max(maximum, neighbor_count[8]);
              maximum = fast_max(maximum, neighbor_count[9]);
              current = N_SPECIES - (maximum & 0x0F);
            }
          }

          total_count[current] += 1;
          next[c] = current;
        }
      }
    }

    /* Update the maximums array */
    for (int i = 1; i <= N_SPECIES; ++i) {
      if (maximums[i].count < total_count[i]) {
        maximums[i].count = total_count[i];
        maximums[i].generation = g;
      }

      total_count[i] = 0;
    }

    /* Swap the previous and next grids */
    unsigned char *temp = previous;
    previous = next;
    next = temp;
  }

  // [1 1 1] to [2 2 2]
  // [1 2 2] + [-1 1 0] -> [0 3 2] = [2 1 2]

  /* Stop tracking time */
  time += omp_get_wtime();
  printf("%.1fs\n", time);

  /* Print the maximums */
  for (int i = 1; i <= N_SPECIES; i++) {
    printf("%d %lld %d\n", i, maximums[i].count, maximums[i].generation);
  }
}