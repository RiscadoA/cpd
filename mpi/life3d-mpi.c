#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N_SPECIES 9

unsigned int internal_seed;

struct task {
  MPI_Comm comm;
  int x_neighbors[2];
  int y_neighbors[2];
  int z_neighbors[2];
  int px, py, pz;
  int sx, sy, sz;
};

struct rle {
  int size, capacity;
  unsigned char *data;
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
 * @brief Initializes the task struct for a given rank, task count and N.
 * @param rank Rank of the task.
 * @param size Total number of tasks.
 * @param N Side length of the grid.
 * @param task Task struct to be initialized.
 */
void init_task(int rank, int size, int N, struct task *task) {
  // Find out how many times we should divide each dimension of the grid.
  int dims[3] = {0, 0, 0};
  MPI_Dims_create(size, 3, dims);

  // Create a cartesian communicator.
  MPI_Cart_create(MPI_COMM_WORLD, 3, dims, (int[]){1, 1, 1}, 0, &task->comm);

  // Find the coordinates of the task.
  int coords[3];
  MPI_Cart_coords(task->comm, rank, 3, coords);

  // Find the ranks of the neighbor tasks.
  MPI_Cart_rank(task->comm, (int[]){coords[0] - 1, coords[1], coords[2]}, &task->x_neighbors[0]);
  MPI_Cart_rank(task->comm, (int[]){coords[0] + 1, coords[1], coords[2]}, &task->x_neighbors[1]);
  MPI_Cart_rank(task->comm, (int[]){coords[0], coords[1] - 1, coords[2]}, &task->y_neighbors[0]);
  MPI_Cart_rank(task->comm, (int[]){coords[0], coords[1] + 1, coords[2]}, &task->y_neighbors[1]);
  MPI_Cart_rank(task->comm, (int[]){coords[0], coords[1], coords[2] - 1}, &task->z_neighbors[0]);
  MPI_Cart_rank(task->comm, (int[]){coords[0], coords[1], coords[2] + 1}, &task->z_neighbors[1]);

  // Find the position and size of the task.
  task->px = (coords[0] * N) / dims[0];
  task->py = (coords[1] * N) / dims[1];
  task->pz = (coords[2] * N) / dims[2];
  task->sx = ((coords[0] + 1) * N) / dims[0] - task->px;
  task->sy = ((coords[1] + 1) * N) / dims[1] - task->py;
  task->sz = ((coords[2] + 1) * N) / dims[2] - task->pz;
}

/**
 * @brief Allocate memory a 3D GOL grid.
 * @param sx Side length of the grid along the x axis.
 * @param sy Side length of the grid along the y axis.
 * @param sz Side length of the grid along the z axis.
 */
unsigned char *alloc_grid(int sx, int sy, int sz) {
  return (unsigned char *)malloc((size_t)(sx + 2) * (size_t)(sy + 2) * (size_t)(sz + 2) *
                                 sizeof(unsigned char));
}

/**
 * @brief Allocates memory for a RLE string, assuming the worst case scenario.
 * @param cells Number of cells to be stored.
 * @return RLE.
 */
struct rle alloc_rle(int cells) {
  struct rle rle;
  rle.size = 0;
  rle.capacity = 2 * cells;
  rle.data = (unsigned char *)malloc((size_t)rle.capacity * sizeof(unsigned char));
  return rle;
}

// Map a 3D coordinate to a linear index in the grid array.
#define linear_from_3d(t, x, y, z) ((x) * ((t).sy + 2) * ((t).sz + 2) + (y) * ((t).sz + 2) + (z))

// Read the value of a neighbor cell, given the linear index of the current cell and the
// displacement in the x, y and z axes.
#define read_neighbor(grid, t, c, dx, dy, dz) grid[(c) + linear_from_3d((t), (dx), (dy), (dz))]

// We use a double !! to convert the species count to a boolean value (0 or 1).
#define is_neighbor_alive(grid, t, c, dx, dy, dz)                                                  \
  !!read_neighbor((grid), (t), (c), (dx), (dy), (dz))

// We use this macro to compute the maximum of two numbers without branching.
// We decided to use it after discovering during profiling that the branch
// miss rate was very high in the original code.
#define fast_max(x, y) ((x) - (((x) - (y)) & ((x) - (y)) >> 31))

/**
 * @brief Copies a part of a grid to a RLE string.
 * @param task Task struct.
 * @param rle RLE string to be copied to.
 * @param grid Grid to be copied.
 * @param px Starting x coordinate.
 * @param py Starting y coordinate.
 * @param pz Starting z coordinate.
 * @param sx Length of the sub-grid along the x axis.
 * @param sy Length of the sub-grid along the y axis.
 * @param sz Length of the sub-grid along the z axis.
 */
void copy_to_rle(const struct task *task, struct rle *rle, const unsigned char *grid, int px,
                 int py, int pz, int sx, int sy, int sz) {
  unsigned char count = 0;
  rle->size = 0;

  for (int x = px; x < px + sx; ++x) {
    for (int y = py; y < py + sy; ++y) {
      for (int z = pz; z < pz + sz; ++z) {
        unsigned char species = grid[linear_from_3d(*task, x, y, z)];
        if (species) {
          if (count) {
            rle->data[rle->size++] = 0;
            rle->data[rle->size++] = count;
            count = 0;
          }
          rle->data[rle->size++] = species;
        } else {
          ++count;
          if (count == 255) {
            rle->data[rle->size++] = 0;
            rle->data[rle->size++] = count;
            count = 0;
          }
        }
      }
    }
  }

  // Remaining dead cells are not written to the RLE string.
  // The receiver knows the size of the sub-grid, so it can infer the number of dead cells.
}

/**
 * Copies a RLE string to a grid.
 *
 * @param task Task struct.
 * @param rle RLE string to be copied from.
 * @param grid Grid to be copied to.
 * @param px Starting x coordinate.
 * @param py Starting y coordinate.
 * @param pz Starting z coordinate.
 * @param sx Length of the sub-grid along the x axis.
 * @param sy Length of the sub-grid along the y axis.
 * @param sz Length of the sub-grid along the z axis.
 */
void copy_from_rle(const struct task *task, const struct rle *rle, unsigned char *grid, int px,
                   int py, int pz, int sx, int sy, int sz) {
  int x = px, y = py, z = pz;

  for (int index = 0; index < rle->size; ++index) {
    unsigned char species = rle->data[index];
    if (species) {
      grid[linear_from_3d(*task, x, y, z)] = species;
      if (++z == pz + sz) {
        z = pz;
        if (++y == py + sy) {
          y = py;
          ++x;
        }
      }
    } else {
      unsigned char count = rle->data[++index];
      for (int i = 0; i < count; ++i) {
        grid[linear_from_3d(*task, x, y, z)] = 0;
        if (++z == pz + sz) {
          z = pz;
          if (++y == py + sy) {
            y = py;
            ++x;
          }
        }
      }
    }
  }

  // Fill the remaining cells with dead cells.
  for (; x < px + sx; ++x) {
    for (; y < py + sy; ++y) {
      for (; z < pz + sz; ++z) {
        grid[linear_from_3d(*task, x, y, z)] = 0;
      }
      z = pz;
    }
    y = py;
  }
}

/**
 * @brief Randomize the grid with a given density.
 * @param grid Grid to be randomized.
 * @param N Side length of the grid.
 * @param density Density of the grid.
 * @param input_seed Seed for the random number generator.
 */
void randomize_grid(unsigned char *grid, const struct task *task, int N, float density,
                    int input_seed) {
  init_random(input_seed);
  for (int x = 0; x < N; x++)
    for (int y = 0; y < N; y++)
      for (int z = 0; z < N; z++) {
        unsigned char species = 0;
        if (get_random() < density) {
          species = (unsigned char)(get_random() * N_SPECIES) + 1;
        }

        if (x >= task->px && x < task->px + task->sx && y >= task->py && y < task->py + task->sy &&
            z >= task->pz && z < task->pz + task->sz) {
          grid[linear_from_3d(*task, x - task->px + 1, y - task->py + 1, z - task->pz + 1)] =
              species;
        }
      }
}

int main(int argc, char **argv) {
  if (argc != 5) {
    fprintf(stderr, "Usage: %s <generations> <side> <density> <seed>\n", argv[0]);
    return 1;
  }

  int generations = atoi(argv[1]);
  int totalN = atoi(argv[2]);
  float density = (float)atof(argv[3]);
  int seed = atoi(argv[4]);

  if (totalN <= 0 || density < 0 || density > 1) {
    fprintf(stderr, "Invalid arguments\n");
    return 1;
  }

  /* Initialize MPI */
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    fprintf(stderr, "Failed to initialize MPI\n");
    return 1;
  }

  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  /* When debugging redirect stderr to a file. */
#ifdef DEBUG
  char filename[100];
  sprintf(filename, "err_%d.txt", rank);
  freopen(filename, "w", stderr);
#endif

  /* Find out who we are. */
  struct task task;
  init_task(rank, size, totalN, &task);
#ifdef DEBUG
  fprintf(stderr, "Local rank: %d\n", rank);
  fprintf(stderr, "Local position: %d %d %d\n", task.px, task.py, task.pz);
  fprintf(stderr, "Local size: %d %d %d\n", task.sx, task.sy, task.sz);
  fprintf(stderr, "Neighbors:\n");
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        fprintf(stderr, "%d ", task.neighbors[i][j][k]);
      }
      fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
  }
  fflush(stderr);
#endif

  /* Initialize the grids */
  unsigned char *previous = alloc_grid(task.sx, task.sy, task.sz);
  unsigned char *next = alloc_grid(task.sx, task.sy, task.sz);
  randomize_grid(previous, &task, totalN, density, seed);

  /* Initialize the buffers we'll be using to communicate */
  struct rle send_rle_z[2] = {alloc_rle(task.sx * task.sy), alloc_rle(task.sx * task.sy)};
  struct rle recv_rle_z[2] = {alloc_rle(task.sx * task.sy), alloc_rle(task.sx * task.sy)};
  MPI_Request recv_req_z[2];

  struct rle send_rle_y[2] = {alloc_rle(task.sx * (task.sz + 2)),
                              alloc_rle(task.sx * (task.sz + 2))};
  struct rle recv_rle_y[2] = {alloc_rle(task.sx * (task.sz + 2)),
                              alloc_rle(task.sx * (task.sz + 2))};
  MPI_Request recv_req_y[2];

  struct rle send_rle_x[2] = {alloc_rle((task.sy + 2) * (task.sz + 2)),
                              alloc_rle((task.sy + 2) * (task.sz + 2))};
  struct rle recv_rle_x[2] = {alloc_rle((task.sy + 2) * (task.sz + 2)),
                              alloc_rle((task.sy + 2) * (task.sz + 2))};
  MPI_Request recv_req_x[2];

  MPI_Request send_req[6];
  MPI_Status status[2];

  /* Count how many of each species is alive in the first generation */
  unsigned long long *history = (unsigned long long *)calloc(
      sizeof(unsigned long long), (size_t)((generations + 1) * (N_SPECIES + 1)));
  for (int x = 1; x <= task.sx; ++x) {
    for (int y = 1; y <= task.sy; ++y) {
      for (int z = 1; z <= task.sz; ++z) {
        history[previous[linear_from_3d(task, x, y, z)]] += 1;
      }
    }
  }

  /* Wait for all tasks to be ready, then start the timer */
  MPI_Barrier(MPI_COMM_WORLD);
  double time = -MPI_Wtime();
  double update_time = 0.0;
  double wait_recv_time = 0.0;
  double wait_send_time = 0.0;
  double barrier_time = 0.0;

  /* Run the simulation */
#pragma omp parallel
  for (int g = 1; g <= generations; g++) {
#pragma omp single
    {
      /**
       * README:
       *
       * We keep a 1-cell border around the grid, which stores the border cells from the
       * neighboring tasks. This is done to avoid having to communicate the whole grid
       * every iteration.
       */

      wait_recv_time = -MPI_Wtime();

      /* Stretch the grid in the z axis */
      copy_to_rle(&task, &send_rle_z[0], previous, 1, 1, 1, task.sx, task.sy, 1);
      MPI_Isend(send_rle_z[0].data, send_rle_z[0].size, MPI_UNSIGNED_CHAR, task.z_neighbors[0], 0,
                task.comm, &send_req[0]);
      copy_to_rle(&task, &send_rle_z[1], previous, 1, 1, task.sz, task.sx, task.sy, 1);
      MPI_Isend(send_rle_z[1].data, send_rle_z[1].size, MPI_UNSIGNED_CHAR, task.z_neighbors[1], 1,
                task.comm, &send_req[1]);

      MPI_Irecv(recv_rle_z[0].data, recv_rle_z[0].capacity, MPI_UNSIGNED_CHAR, task.z_neighbors[0],
                1, task.comm, &recv_req_z[0]);
      MPI_Irecv(recv_rle_z[1].data, recv_rle_z[1].capacity, MPI_UNSIGNED_CHAR, task.z_neighbors[1],
                0, task.comm, &recv_req_z[1]);

      MPI_Waitall(2, recv_req_z, status);
      MPI_Get_count(&status[0], MPI_UNSIGNED_CHAR, &recv_rle_z[0].size);
      MPI_Get_count(&status[1], MPI_UNSIGNED_CHAR, &recv_rle_z[1].size);
      copy_from_rle(&task, &recv_rle_z[0], previous, 1, 1, 0, task.sx, task.sy, 1);
      copy_from_rle(&task, &recv_rle_z[1], previous, 1, 1, task.sz + 1, task.sx, task.sy, 1);

      /* Stretch the grid in the y axis */
      copy_to_rle(&task, &send_rle_y[0], previous, 1, 1, 0, task.sx, 1, task.sz + 2);
      MPI_Isend(send_rle_y[0].data, send_rle_y[0].size, MPI_UNSIGNED_CHAR, task.y_neighbors[0], 2,
                task.comm, &send_req[2]);
      copy_to_rle(&task, &send_rle_y[1], previous, 1, task.sy, 0, task.sx, 1, task.sz + 2);
      MPI_Isend(send_rle_y[1].data, send_rle_y[1].size, MPI_UNSIGNED_CHAR, task.y_neighbors[1], 3,
                task.comm, &send_req[3]);

      MPI_Irecv(recv_rle_y[0].data, recv_rle_y[0].capacity, MPI_UNSIGNED_CHAR, task.y_neighbors[0],
                3, task.comm, &recv_req_y[0]);
      MPI_Irecv(recv_rle_y[1].data, recv_rle_y[1].capacity, MPI_UNSIGNED_CHAR, task.y_neighbors[1],
                2, task.comm, &recv_req_y[1]);

      MPI_Waitall(2, recv_req_y, status);
      MPI_Get_count(&status[0], MPI_UNSIGNED_CHAR, &recv_rle_y[0].size);
      MPI_Get_count(&status[1], MPI_UNSIGNED_CHAR, &recv_rle_y[1].size);
      copy_from_rle(&task, &recv_rle_y[0], previous, 1, 0, 0, task.sx, 1, task.sz + 2);
      copy_from_rle(&task, &recv_rle_y[1], previous, 1, task.sy + 1, 0, task.sx, 1, task.sz + 2);

      /* Stretch the grid in the x axis */
      copy_to_rle(&task, &send_rle_x[0], previous, 1, 0, 0, 1, task.sy + 2, task.sz + 2);
      MPI_Isend(send_rle_x[0].data, send_rle_x[0].size, MPI_UNSIGNED_CHAR, task.x_neighbors[0], 4,
                task.comm, &send_req[4]);
      copy_to_rle(&task, &send_rle_x[1], previous, task.sx, 0, 0, 1, task.sy + 2, task.sz + 2);
      MPI_Isend(send_rle_x[1].data, send_rle_x[1].size, MPI_UNSIGNED_CHAR, task.x_neighbors[1], 5,
                task.comm, &send_req[5]);

      MPI_Irecv(recv_rle_x[0].data, recv_rle_x[0].capacity, MPI_UNSIGNED_CHAR, task.x_neighbors[0],
                5, task.comm, &recv_req_x[0]);
      MPI_Irecv(recv_rle_x[1].data, recv_rle_x[1].capacity, MPI_UNSIGNED_CHAR, task.x_neighbors[1],
                4, task.comm, &recv_req_x[1]);

      MPI_Waitall(2, recv_req_x, status);
      MPI_Get_count(&status[0], MPI_UNSIGNED_CHAR, &recv_rle_x[0].size);
      MPI_Get_count(&status[1], MPI_UNSIGNED_CHAR, &recv_rle_x[1].size);
      copy_from_rle(&task, &recv_rle_x[0], previous, 0, 0, 0, 1, task.sy + 2, task.sz + 2);
      copy_from_rle(&task, &recv_rle_x[1], previous, task.sx + 1, 0, 0, 1, task.sy + 2,
                    task.sz + 2);

      /* Wait for the data to be sent */
      wait_recv_time += MPI_Wtime();
      wait_send_time = -MPI_Wtime();
      MPI_Waitall(6, send_req, MPI_STATUSES_IGNORE);
      wait_send_time += MPI_Wtime();

      update_time = -MPI_Wtime();

#ifdef DEBUG
      /* Print entire state of grid */
      fprintf(stderr, "Generation %d's previous state:\n", g);
      for (int x = 0; x <= task.sx + 1; ++x) {
        for (int y = 0; y <= task.sy + 1; ++y) {
          for (int z = 0; z <= task.sz + 1; ++z) {
            fprintf(stderr, "%d ", previous[linear_from_3d(task, x, y, z)]);
          }
          fprintf(stderr, "\n");
        }
        fprintf(stderr, "\n");
      }
      fflush(stderr);
#endif
    }

/* Update the cells */
#pragma omp for schedule(static)
    for (int x = 1; x <= task.sx; ++x) {
      for (int y = 1; y <= task.sy; ++y) {
#pragma omp simd
        for (int z = 1; z <= task.sz; ++z) {
          int c = linear_from_3d(task, x, y, z);
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
            live_neighbors += is_neighbor_alive(previous, task, c, 0, 0, -1);
            live_neighbors += is_neighbor_alive(previous, task, c, 0, 0, 1);
            live_neighbors += is_neighbor_alive(previous, task, c, -1, -1, -1);
            live_neighbors += is_neighbor_alive(previous, task, c, -1, -1, 0);
            live_neighbors += is_neighbor_alive(previous, task, c, -1, -1, 1);
            live_neighbors += is_neighbor_alive(previous, task, c, -1, 0, -1);
            live_neighbors += is_neighbor_alive(previous, task, c, -1, 0, 0);
            live_neighbors += is_neighbor_alive(previous, task, c, -1, 0, 1);
            live_neighbors += is_neighbor_alive(previous, task, c, -1, 1, -1);
            live_neighbors += is_neighbor_alive(previous, task, c, -1, 1, 0);
            live_neighbors += is_neighbor_alive(previous, task, c, -1, 1, 1);
            live_neighbors += is_neighbor_alive(previous, task, c, 0, -1, -1);
            live_neighbors += is_neighbor_alive(previous, task, c, 0, -1, 0);
            live_neighbors += is_neighbor_alive(previous, task, c, 0, -1, 1);
            live_neighbors += is_neighbor_alive(previous, task, c, 0, 1, -1);
            live_neighbors += is_neighbor_alive(previous, task, c, 0, 1, 0);
            live_neighbors += is_neighbor_alive(previous, task, c, 0, 1, 1);
            live_neighbors += is_neighbor_alive(previous, task, c, 1, -1, -1);
            live_neighbors += is_neighbor_alive(previous, task, c, 1, -1, 0);
            live_neighbors += is_neighbor_alive(previous, task, c, 1, -1, 1);
            live_neighbors += is_neighbor_alive(previous, task, c, 1, 0, -1);
            live_neighbors += is_neighbor_alive(previous, task, c, 1, 0, 0);
            live_neighbors += is_neighbor_alive(previous, task, c, 1, 0, 1);
            live_neighbors += is_neighbor_alive(previous, task, c, 1, 1, -1);
            live_neighbors += is_neighbor_alive(previous, task, c, 1, 1, 0);
            live_neighbors += is_neighbor_alive(previous, task, c, 1, 1, 1);
            if (live_neighbors <= 4 || live_neighbors > 13) {
              current = 0;
            }
          } else {
            // Same as above, but we collect the counts for each species.
            int neighbor_count[N_SPECIES + 1] = {0};
            neighbor_count[read_neighbor(previous, task, c, 0, 0, -1)] += 1;
            neighbor_count[read_neighbor(previous, task, c, 0, 0, 1)] += 1;
            neighbor_count[read_neighbor(previous, task, c, -1, -1, -1)] += 1;
            neighbor_count[read_neighbor(previous, task, c, -1, -1, 0)] += 1;
            neighbor_count[read_neighbor(previous, task, c, -1, -1, 1)] += 1;
            neighbor_count[read_neighbor(previous, task, c, -1, 0, -1)] += 1;
            neighbor_count[read_neighbor(previous, task, c, -1, 0, 0)] += 1;
            neighbor_count[read_neighbor(previous, task, c, -1, 0, 1)] += 1;
            neighbor_count[read_neighbor(previous, task, c, -1, 1, -1)] += 1;
            neighbor_count[read_neighbor(previous, task, c, -1, 1, 0)] += 1;
            neighbor_count[read_neighbor(previous, task, c, -1, 1, 1)] += 1;
            neighbor_count[read_neighbor(previous, task, c, 0, -1, -1)] += 1;
            neighbor_count[read_neighbor(previous, task, c, 0, -1, 0)] += 1;
            neighbor_count[read_neighbor(previous, task, c, 0, -1, 1)] += 1;
            neighbor_count[read_neighbor(previous, task, c, 0, 1, -1)] += 1;
            neighbor_count[read_neighbor(previous, task, c, 0, 1, 0)] += 1;
            neighbor_count[read_neighbor(previous, task, c, 0, 1, 1)] += 1;
            neighbor_count[read_neighbor(previous, task, c, 1, -1, -1)] += 1;
            neighbor_count[read_neighbor(previous, task, c, 1, -1, 0)] += 1;
            neighbor_count[read_neighbor(previous, task, c, 1, -1, 1)] += 1;
            neighbor_count[read_neighbor(previous, task, c, 1, 0, -1)] += 1;
            neighbor_count[read_neighbor(previous, task, c, 1, 0, 0)] += 1;
            neighbor_count[read_neighbor(previous, task, c, 1, 0, 1)] += 1;
            neighbor_count[read_neighbor(previous, task, c, 1, 1, -1)] += 1;
            neighbor_count[read_neighbor(previous, task, c, 1, 1, 0)] += 1;
            neighbor_count[read_neighbor(previous, task, c, 1, 1, 1)] += 1;

            // To get the number of live neighbors, we subtract the number of dead neighbors
            // from the total number of neighbors.
            int live_neighbors = 26 - neighbor_count[0];
            if (live_neighbors >= 7 && live_neighbors <= 10) {
#pragma GCC unroll 9
              for (int i = 1; i <= N_SPECIES; ++i) {
                // We store the species count in the lower 4 bits of the neighbor count, in order
                // to use a branchless max. Notice that we store the species as N_SPECIES - i, so
                // that the species with a lower number have higher priority in case of a tie.
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

          history[g * (N_SPECIES + 1) + current] += 1;
          next[c] = current;
        }
      }
    }

#pragma omp single
    {
      update_time += MPI_Wtime();
      barrier_time = -MPI_Wtime();

      /* Wait for other tasks to finish the generation */
      MPI_Barrier(task.comm);

      barrier_time += MPI_Wtime();

      fprintf(stderr,
              "Generation %d rank %d: update=%.6f wait_recv=%.6f wait_send=%.6f "
              "barrier_time=%.6f\n",
              g, rank, update_time, wait_recv_time, wait_send_time, barrier_time);

      /* Swap the previous and next grids */
      unsigned char *temp = previous;
      previous = next;
      next = temp;
    }
  }

  /* Reduce the history to the root task */
  if (rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, history, (generations + 1) * (N_SPECIES + 1), MPI_UNSIGNED_LONG_LONG,
               MPI_SUM, 0, (task).comm);
  } else {
    MPI_Reduce(history, NULL, (generations + 1) * (N_SPECIES + 1), MPI_UNSIGNED_LONG_LONG, MPI_SUM,
               0, (task).comm);
  }

  /* If we're the root task, find the maximums */
  struct {
    unsigned long long count;
    int generation;
  } maximums[N_SPECIES + 1] = {0};

  if (rank == 0) {
    for (int g = 0; g <= generations; g++) {
      for (int i = 1; i <= N_SPECIES; i++) {
        if (history[g * (N_SPECIES + 1) + i] > maximums[i].count) {
          maximums[i].count = history[g * (N_SPECIES + 1) + i];
          maximums[i].generation = g;
        }
      }
    }
  }

  /* Wait for all tasks to finish, then stop the timer */
  MPI_Barrier(task.comm);
  time += MPI_Wtime();
  if (rank == 0) {
    fprintf(stderr, "%.1f\n", time);

    /* Print the maximums */
    for (int i = 1; i <= N_SPECIES; i++) {
      printf("%d %lld %d\n", i, maximums[i].count, maximums[i].generation);
    }
  }

  /* Terminate MPI */
  MPI_Finalize();

  /* Free memory */
  free(previous);
  free(next);

  for (int i = 0; i < 2; ++i) {
    free(send_rle_x[i].data);
    free(send_rle_y[i].data);
    free(send_rle_z[i].data);
    free(recv_rle_x[i].data);
    free(recv_rle_y[i].data);
    free(recv_rle_z[i].data);
  }
}
