#include "rng.h"
#include <cmath>
#include <initializer_list>

namespace broadside::sim {
std::uint32_t Rng::nextU32() {
    state_ += 0x6d2b79f5u;
    std::uint32_t t = state_;
    t = (t ^ (t >> 15)) * (t | 1u);
    t ^= t + ((t ^ (t >> 7)) * (t | 61u));
    return t ^ (t >> 14);
}
double Rng::next() {
    return static_cast<double>(nextU32()) / 4294967296.0;
}
double Rng::range(double lo, double hi) {
    return lo + next() * (hi - lo);
}
int Rng::integer(int lo, int hi) {
    return static_cast<int>(std::floor(lo + next() * static_cast<double>(hi - lo + 1)));
}
std::uint32_t hashSeed(std::initializer_list<std::uint32_t> values) {
    std::uint32_t h = 2166136261u;
    for (auto n : values) {
        h ^= n;
        h *= 16777619u;
    }
    return h;
}
} // namespace broadside::sim
