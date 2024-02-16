#include <iostream>

static constexpr int NSpecies = 9;

/// @brief Generates random numbers.
class Random {
public:
  /// @brief Constructs a random number generator with the given seed.
  /// @param seed Seed.
  Random(unsigned int seed) { mSeed = seed + 987654321; }

  /// @brief Returns the next random number.Random
  /// @return Random number.
  float next() {
    int prev = mSeed;
    mSeed ^= (mSeed << 13);
    mSeed ^= (mSeed >> 17);
    mSeed ^= (mSeed << 5);
    return 0.5 + 0.2328306e-09 * (prev + (int)mSeed);
  }

private:
  unsigned int mSeed;
};

/// @brief Stores the parsed command line arguments.
struct Arguments {
  int generations;
  int side;
  float density;
  int seed;
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
  std::size_t side() const { return mSide; }

  /// @brief Returns the cell at the given position.
  /// @param x X coordinate of the cell.
  /// @param y Y coordinate of the cell.
  /// @param z Z coordinate of the cell.
  /// @return Cell at the given position.
  unsigned char &cell(int x, int y, int z) { return this->cell(x, y, z, mActive); }

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
            cell(x, y, z) = static_cast<int>(random.next() * NSpecies) + 1;
          } else {
            cell(x, y, z) = 0;
          }
        }
      }
    }
  }

  /// @brief Advances the world by one generation.
  void advance() {
    int counter[NSpecies + 1];

    for (int z = 0; z < mSide; ++z) {
      for (int y = 0; y < mSide; ++y) {
        for (int x = 0; x < mSide; ++x) {
          // Count the number of neighbors.
          auto neighbors = 0;
          forEachNeighbour(x, y, z, [&](int nx, int ny, int nz) {
            if (cell(nx, ny, nz, mActive)) {
              ++neighbors;
            }
          });

          // Apply the rules.
          auto &old = cell(x, y, z, mActive);
          auto &next = cell(x, y, z, 1 - mActive);
          if (old) {
            if (neighbors < 5 || neighbors > 13) {
              next = 0;
            }
          } else if (neighbors >= 7 && neighbors <= 10) {
            for (int i = 0; i < 9; i++) {
              counter[i] = 0;
            }

            // Count each species in the neighborhood.
            forEachNeighbour(
                x, y, z, [&](int nx, int ny, int nz) { counter[cell(nx, ny, nz, mActive)] += 1; });

            // Set the cell to the most common species.
            next = 1;
            for (int i = 2; i <= NSpecies; i++) {
              if (counter[i] > counter[next]) {
                next = i;
              }
            }
          }
        }
      }
    }

    mActive = 1 - mActive;
  }

private:
  int mActive;
  std::size_t mSide;
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
    arguments.seed = std::stoull(argv[4]);
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

  return 0;
}
