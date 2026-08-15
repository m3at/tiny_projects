#include "geometry.h"
#include <cmath>

namespace broadside::sim {
double fsin(double v) {
    return static_cast<double>(static_cast<float>(std::sin(v)));
}
double fcos(double v) {
    return static_cast<double>(static_cast<float>(std::cos(v)));
}
double fatan2(double y, double x) {
    return static_cast<double>(static_cast<float>(std::atan2(y, x)));
}
double length(Vec2 v) {
    return std::sqrt(v.x * v.x + v.z * v.z);
}
double wrapAngle(double a) {
    if (a >= -Pi && a <= Pi)
        return a;
    a = std::fmod(a + Pi, 2 * Pi);
    if (a < 0)
        a += 2 * Pi;
    return a - Pi;
}
Vec2 localToWorld(const Vec2 &o, double c, double s, Coord p) {
    const double lx = p.dx * CellSize, lz = p.dz * CellSize;
    return {o.x + lx * c - lz * s, o.z + lx * s + lz * c};
}
} // namespace broadside::sim
