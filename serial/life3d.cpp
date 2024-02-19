#include <iomanip>
#include <iostream>
#include <omp.h>

static constexpr unsigned char NSpecies = 9;

/// @brief Generates random numbers.
class Random {
public:
  /// @brief Constructs a random number generator with the given seed.
  /// @param seed Seed.
  Random(unsigned int seed) { mSeed = seed + 987654321; }

  /// @brief Returns the next random number.Random
  /// @return Random number.
  float next() {
    auto prev = static_cast<int>(mSeed);
    mSeed ^= (mSeed << 13);
    mSeed ^= (mSeed >> 17);
    mSeed ^= (mSeed << 5);
    return static_cast<float>(0.5 +
                              0.2328306e-09 * static_cast<double>(prev + static_cast<int>(mSeed)));
  }

private:
  unsigned int mSeed;
};

/// @brief Stores the parsed command line arguments.
struct Arguments {
  int generations;
  int side;
  float density;
  unsigned int seed;
};

struct Statistics {
  unsigned long long count;
  int generation;
};

/// @brief Stores a 3D GOL World.
class Life3D {
public:
  ~Life3D() { delete[] mCells; }

  /// @brief Constructs a 3D GOL world with the given side length.
  /// @param side Side length of the 3D world.
  Life3D(int side) : mSide{side} {
    auto bigSide = static_cast<std::size_t>(side);
    mCells = new unsigned char[bigSide * bigSide * bigSide * 2];
  }

  /// @brief Returns the side length of the 3D world.
  int side() const { return mSide; }

  /// @brief Returns the cell at the given position.
  /// @param x X coordinate of the cell.
  /// @param y Y coordinate of the cell.
  /// @param z Z coordinate of the cell.
  /// @param buffer Buffer to read from.
  /// @return Cell at the given position.
  unsigned char &cell(int x, int y, int z, int buffer) {
    return mCells[x + y * mSide + z * mSide * mSide + buffer * mSide * mSide * mSide];
  }

  void forEachNeighbour(int x, int y, int z, auto lambda) {
    for (auto dz : {-1, 0, 1}) {
      for (auto dy : {-1, 0, 1}) {
        for (auto dx : {-1, 0, 1}) {
          if (dx == 0 && dy == 0 && dz == 0) {
            continue;
          }

          // Wrap around the edges.
          auto nx = (x + dx + mSide) % mSide;
          auto ny = (y + dy + mSide) % mSide;
          auto nz = (z + dz + mSide) % mSide;
          lambda(nx, ny, nz);
        }
      }
    }
  }

  /// @brief Initializes the world with random cells.
  /// @param density Density of the cells.
  /// @param seed Seed for the random number generator.
  void randomize(float density, unsigned int seed) {
    Random random{seed};
    for (int z = 0; z < mSide; ++z) {
      for (int y = 0; y < mSide; ++y) {
        for (int x = 0; x < mSide; ++x) {
          if (random.next() < density) {
            cell(x, y, z, 0) = static_cast<unsigned char>(random.next() * NSpecies) + 1;
          } else {
            cell(x, y, z, 0) = 0;
          }
        }
      }
    }
  }

  /// @brief Calculates the next generation from the current one.
  void update(int generation, Statistics stats[NSpecies + 1]) {
    unsigned long long counts[NSpecies + 1]{0};
    int active = generation % 2;

    for (int z = 0; z < mSide; ++z) {
      for (int y = 0; y < mSide; ++y) {
        for (int x = 0; x < mSide; ++x) {
          // Count the number of neighbors.
          auto neighbors = 0;
          forEachNeighbour(x, y, z, [&](int nx, int ny, int nz) {
            if (cell(nx, ny, nz, active)) {
              ++neighbors;
            }
          });

          auto &old = cell(x, y, z, active);
          counts[old] += 1;

          // Apply the rules.
          auto &next = cell(x, y, z, 1 - active);
          if (old) {
            next = (neighbors < 5 || neighbors > 13) ? 0 : old;
          } else if (neighbors >= 7 && neighbors <= 10) {
            int counter[NSpecies + 1]{0};

            // Count each species in the neighborhood.
            forEachNeighbour(
                x, y, z, [&](int nx, int ny, int nz) { counter[cell(nx, ny, nz, active)] += 1; });

            // Set the cell to the most common species.
            next = 1;
            for (unsigned char i = 2; i <= NSpecies; i++) {
              if (counter[i] > counter[next]) {
                next = i;
              }
            }
          } else {
            next = 0;
          }
        }
      }
    }

    for (int i = 1; i <= NSpecies; i++) {
      if (counts[i] > stats[i].count) {
        stats[i] = {counts[i], generation};
      }
    }
  }

private:
  int mSide;
  unsigned char *mCells;
};

/// @brief Parses the command line arguments and stores them in the given
/// Arguments.
/// @param argc Command line argument count.
/// @param argv Command line arguments.
/// @param arguments Arguments to store the parsed values in.
/// @return Whether the arguments were parsed successfully.
static bool parseArguments(int argc, char **argv, Arguments &arguments) {
  if (argc != 5) {
    std::cerr << "Usage: " << argv[0] << " <generations> <cube side> <density> <seed>" << std::endl;
    return false;
  }

  try {
    arguments.generations = std::stoi(argv[1]);
    arguments.side = std::stoi(argv[2]);
    arguments.density = std::stof(argv[3]);
    arguments.seed = static_cast<unsigned int>(std::stoul(argv[4]));
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Could not parse arguments: " << e.what() << std::endl;
    return false;
  }
}

int main(int argc, char **argv) {
  Arguments arguments;
  if (!parseArguments(argc, argv, arguments)) {
    std::cerr << "Could not parse arguments" << std::endl;
    return 1;
  }

  Statistics stats[NSpecies + 1];
  for (int i = 0; i <= NSpecies; i++) {
    stats[i] = {0, 0};
  }

  Life3D life{arguments.side};
  life.randomize(arguments.density, arguments.seed);
  double time = -omp_get_wtime();

  for (int generation = 0; generation < arguments.generations; ++generation) {
    life.update(generation, stats);
  }

  time += omp_get_wtime();
  std::cerr << std::fixed << std::setprecision(1) << time << "s" << std::endl;

  for (int i = 1; i <= NSpecies; i++) {
    std::cout << i << " " << stats[i].count << " " << stats[i].generation << std::endl;
  }

  return 0;
}
