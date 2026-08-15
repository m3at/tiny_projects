#pragma once
#include "types.h"

namespace broadside::sim {
constexpr double Pi = 3.14159265358979323846;
double fsin(double value);
double fcos(double value);
double fatan2(double y, double x);
double length(Vec2 v);
double wrapAngle(double angle);
Vec2 localToWorld(const Vec2 &origin, double cosHeading, double sinHeading, Coord cell);
} // namespace broadside::sim
