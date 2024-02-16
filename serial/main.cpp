#include <iostream>

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0]
              << " <generations> <cube side> <density> <seed>" << std::endl;
    return 1;
  }
}
