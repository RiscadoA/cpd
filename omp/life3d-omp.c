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

// Map a 3D coordinate to a linear index in the grid array.
#define linear_from_3d(S, x, y, z) ((x) * (S) * (S) + (y) * (S) + (z))

// Read the value of a neighbor cell, given the linear index of the current cell and the
// displacement in the x, y and z axes.
#define read_neighbor(grid, S, c, dx, dy, dz) grid[(c) + linear_from_3d((S), (dx), (dy), (dz))]

// We use a double !! to convert the species count to a boolean value (0 or 1).
#define is_neighbor_alive(grid, S, c, dx, dy, dz)                                                  \
  !!read_neighbor((grid), (S), (c), (dx), (dy), (dz))

// We use this macro to compute the maximum of two numbers without branching.
// We decided to use it after discovering during profiling that the branch
// miss rate was very high in the original code.
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
    /**
     * README:
     *
     * To reduce the complexity of the boundary conditions (wrap around), instead of performing
     * expensive modulus and division operations every neighbor access, we add a border of 1 cell
     * around the grid. This border is initialized with the same values as the opposite side of the
     * grid, so that the boundary conditions are satisfied.
     *
     * The following three loops correspond to that initialization.
     */

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
#pragma omp parallel for schedule(static)
    for (int x = 1; x <= N; ++x) {
      for (int y = 1; y <= N; ++y) {
        for (int z = 1; z <= N; ++z) {
          int c = linear_from_3d(N + 2, x, y, z);
          unsigned char current = previous[c];

          /**
           * README:
           *
           * The following code is a manual unrolling of the loop that checks the
           * neighbors of a cell. This is done to ensure we access the grid in the
           * optimal way, cache-wise. We compared the performance with the original
           * loop using #pragma GCC unroll, but we got 30% worse performance.
           *
           * We also omit the line that would check the current cell, as we only want
           * to check the neighbors.
           */

          if (current) {
            int live_neighbors = 0;

            // We first access the neighbors in the same line as the current cell.
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
            // Same as above, but we collect the counts for each species.
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

            // To get the number of live neighbors, we subtract the number of dead neighbors
            // from the total number of neighbors.
            int live_neighbors = 26 - neighbor_count[0];
            if (live_neighbors >= 7 && live_neighbors <= 10) {
#pragma GCC unroll 9
              for (int i = 1; i <= N_SPECIES; ++i) {
                // We store the species count in the lower 4 bits of the neighbor count, in order to
                // use a branchless max. Notice that we store the species as N_SPECIES - i, so that
                // the species with a lower number have higher priority in case of a tie.
                neighbor_count[i] = (neighbor_count[i] << 4) | (N_SPECIES - i);
              }

              // Use a branchless max to find the maximum of the values computed above.
              int maximum = neighbor_count[1];
#pragma GCC unroll 9
              for (int i = 2; i <= N_SPECIES; ++i) {
                maximum = fast_max(maximum, neighbor_count[i]);
              }

              // Extract the species from the lower 4 bits of the maximum value.
              current = N_SPECIES - (maximum & 0x0F);
            }
          }

#pragma omp atomic
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

  /* Stop tracking time */
  time += omp_get_wtime();
  fprintf(stderr, "%.1fs\n", time);

  /* Print the maximums */
  for (int i = 1; i <= N_SPECIES; i++) {
    printf("%d %lld %d\n", i, maximums[i].count, maximums[i].generation);
  }
}