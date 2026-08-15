#pragma once
#include <cstdint>
#include <initializer_list>
#include <vector>

namespace broadside::sim {
class Rng {
  public:
    explicit Rng(std::uint32_t seed) : state_(seed) {}
    std::uint32_t nextU32();
    double next();
    double range(double lo, double hi);
    int integer(int lo, int hi);
    bool chance(double probability) {
        return next() < probability;
    }

  private:
    std::uint32_t state_;
};
std::uint32_t hashSeed(std::initializer_list<std::uint32_t> values);
} // namespace broadside::sim
