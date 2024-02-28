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
  size_t side = (size_t)N;
  return (unsigned char *)malloc(side * side * side * sizeof(unsigned char));
}

/**
 * @brief Free the memory of a 3D GOL grid.
 * @param grid Grid to be freed.
 */
void free_grid(unsigned char *grid) { free(grid); }

/**
 * @brief Randomize the grid with a given density.
 * @param grid Grid to be randomized.
 * @param N Side length of the grid.
 * @param density Density of the grid.
 * @param input_seed Seed for the random number generator.
 */
void randomize_grid(unsigned char *grid, int N, float density, int input_seed) {
  init_random(input_seed);
  for (int x = 0; x < N; x++)
    for (int y = 0; y < N; y++)
      for (int z = 0; z < N; z++)
        if (get_random() < density)
          grid[x * N * N + y * N + z] = (unsigned char)(get_random() * N_SPECIES) + 1;
}

/**
 * @brief Accesses a cell in the grid.
 * @param grid Grid to be accessed.
 * @param N Side length of the grid.
 * @param x X coordinate of the cell.
 * @param y Y coordinate of the cell.
 * @param z Z coordinate of the cell.
 */
#define read_grid(grid, N, x, y, z) grid[x * N * N + y * N + z]

/**
 * @brief Sets a cell in the grid.
 * @param grid Grid to be accessed.
 * @param N Side length of the grid.
 * @param x X coordinate of the cell.
 * @param y Y coordinate of the cell.
 * @param z Z coordinate of the cell.
 * @param value Value to be set.
 */
#define write_grid(grid, N, x, y, z, value) (grid[x * N * N + y * N + z] = value)

#define wrap_around(N, c) ((c + N) % N)

#define linear_from_3d(N, x, y, z)                                                                 \
  (wrap_around(N, x) * N * N + wrap_around(N, y) * N + wrap_around(N, z))

#define read_neighbor(grid, N, x, y, z, dx, dy, dz) grid[linear_from_3d(N, x + dx, y + dy, z + dz)]

#define fast_max(x, y) (x - ((x - y) & (x - y) >> 31))

/**
 * @brief Calculates the next generation of the grid from the current one.
 * @param previous Previous generation of the grid.
 * @param next Next generation of the grid.
 * @param N Side length of the grid.
 * @param total_count Array which keeps the count of each species.
 */
void update_grid(unsigned char *previous, unsigned char *next, int N,
                 long long total_count[N_SPECIES + 1]) {
  for (int x = 0; x < N; x++) {
    for (int y = 0; y < N; y++) {
      for (int z = 0; z < N; z++) {
        unsigned char current = read_grid(previous, N, x, y, z);

        if (current) {
          int live_neighbors = 0;
          live_neighbors += !!read_neighbor(previous, N, x, y, z, 0, 0, -1);
          live_neighbors += !!read_neighbor(previous, N, x, y, z, 0, 0, 1);
          live_neighbors += !!read_neighbor(previous, N, x, y, z, -1, -1, -1);
          live_neighbors += !!read_neighbor(previous, N, x, y, z, -1, -1, 0);
          live_neighbors += !!read_neighbor(previous, N, x, y, z, -1, -1, 1);
          live_neighbors += !!read_neighbor(previous, N, x, y, z, -1, 0, -1);
          live_neighbors += !!read_neighbor(previous, N, x, y, z, -1, 0, 0);
          live_neighbors += !!read_neighbor(previous, N, x, y, z, -1, 0, 1);
          live_neighbors += !!read_neighbor(previous, N, x, y, z, -1, 1, -1);
          live_neighbors += !!read_neighbor(previous, N, x, y, z, -1, 1, 0);
          live_neighbors += !!read_neighbor(previous, N, x, y, z, -1, 1, 1);
          live_neighbors += !!read_neighbor(previous, N, x, y, z, 0, -1, -1);
          live_neighbors += !!read_neighbor(previous, N, x, y, z, 0, -1, 0);
          live_neighbors += !!read_neighbor(previous, N, x, y, z, 0, -1, 1);
          live_neighbors += !!read_neighbor(previous, N, x, y, z, 0, 1, -1);
          live_neighbors += !!read_neighbor(previous, N, x, y, z, 0, 1, 0);
          live_neighbors += !!read_neighbor(previous, N, x, y, z, 0, 1, 1);
          live_neighbors += !!read_neighbor(previous, N, x, y, z, 1, -1, -1);
          live_neighbors += !!read_neighbor(previous, N, x, y, z, 1, -1, 0);
          live_neighbors += !!read_neighbor(previous, N, x, y, z, 1, -1, 1);
          live_neighbors += !!read_neighbor(previous, N, x, y, z, 1, 0, -1);
          live_neighbors += !!read_neighbor(previous, N, x, y, z, 1, 0, 0);
          live_neighbors += !!read_neighbor(previous, N, x, y, z, 1, 0, 1);
          live_neighbors += !!read_neighbor(previous, N, x, y, z, 1, 1, -1);
          live_neighbors += !!read_neighbor(previous, N, x, y, z, 1, 1, 0);
          live_neighbors += !!read_neighbor(previous, N, x, y, z, 1, 1, 1);
          if (live_neighbors <= 4 || live_neighbors > 13) {
            current = 0;
          }
        } else {
          int neighbor_count[N_SPECIES + 1] = {0};
          neighbor_count[read_neighbor(previous, N, x, y, z, 0, 0, -1)] += 1;
          neighbor_count[read_neighbor(previous, N, x, y, z, 0, 0, 1)] += 1;
          neighbor_count[read_neighbor(previous, N, x, y, z, -1, -1, -1)] += 1;
          neighbor_count[read_neighbor(previous, N, x, y, z, -1, -1, 0)] += 1;
          neighbor_count[read_neighbor(previous, N, x, y, z, -1, -1, 1)] += 1;
          neighbor_count[read_neighbor(previous, N, x, y, z, -1, 0, -1)] += 1;
          neighbor_count[read_neighbor(previous, N, x, y, z, -1, 0, 0)] += 1;
          neighbor_count[read_neighbor(previous, N, x, y, z, -1, 0, 1)] += 1;
          neighbor_count[read_neighbor(previous, N, x, y, z, -1, 1, -1)] += 1;
          neighbor_count[read_neighbor(previous, N, x, y, z, -1, 1, 0)] += 1;
          neighbor_count[read_neighbor(previous, N, x, y, z, -1, 1, 1)] += 1;
          neighbor_count[read_neighbor(previous, N, x, y, z, 0, -1, -1)] += 1;
          neighbor_count[read_neighbor(previous, N, x, y, z, 0, -1, 0)] += 1;
          neighbor_count[read_neighbor(previous, N, x, y, z, 0, -1, 1)] += 1;
          neighbor_count[read_neighbor(previous, N, x, y, z, 0, 1, -1)] += 1;
          neighbor_count[read_neighbor(previous, N, x, y, z, 0, 1, 0)] += 1;
          neighbor_count[read_neighbor(previous, N, x, y, z, 0, 1, 1)] += 1;
          neighbor_count[read_neighbor(previous, N, x, y, z, 1, -1, -1)] += 1;
          neighbor_count[read_neighbor(previous, N, x, y, z, 1, -1, 0)] += 1;
          neighbor_count[read_neighbor(previous, N, x, y, z, 1, -1, 1)] += 1;
          neighbor_count[read_neighbor(previous, N, x, y, z, 1, 0, -1)] += 1;
          neighbor_count[read_neighbor(previous, N, x, y, z, 1, 0, 0)] += 1;
          neighbor_count[read_neighbor(previous, N, x, y, z, 1, 0, 1)] += 1;
          neighbor_count[read_neighbor(previous, N, x, y, z, 1, 1, -1)] += 1;
          neighbor_count[read_neighbor(previous, N, x, y, z, 1, 1, 0)] += 1;
          neighbor_count[read_neighbor(previous, N, x, y, z, 1, 1, 1)] += 1;

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
        write_grid(next, N, x, y, z, current);
      }
    }
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

  /* Count how many of each species is alive */
  long long total_count[N_SPECIES + 1] = {0};
  for (int x = 0; x < N; ++x) {
    for (int y = 0; y < N; ++y) {
      for (int z = 0; z < N; ++z) {
        total_count[read_grid(previous, N, x, y, z)] += 1;
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

  /* Start tracking time */
  double time = -omp_get_wtime();

  /* Run the simulation */
  for (int g = 1; g <= generations; g++) {
    update_grid(previous, next, N, total_count);

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

  /* Stop tracking time */
  time += omp_get_wtime();
  printf("%.1fs\n", time);

  /* Print the maximums */
  for (int i = 1; i <= N_SPECIES; i++) {
    printf("%d %lld %d\n", i, maximums[i].count, maximums[i].generation);
  }
}