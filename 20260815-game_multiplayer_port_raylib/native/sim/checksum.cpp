#include "checksum.h"
#include <cmath>
namespace broadside::sim {
std::uint32_t checksum(const Battle &b) {
    std::uint32_t h = 2166136261u;
    auto add = [&](std::uint32_t n) {
        h ^= n;
        h *= 16777619u;
    };
    add(static_cast<std::uint32_t>(b.tickCount()));
    for (const auto &s : b.state()) {
        add(static_cast<std::uint32_t>(std::lround(s.position.x * 4096)));
        add(static_cast<std::uint32_t>(std::lround(s.position.z * 4096)));
        add(static_cast<std::uint32_t>(std::lround(s.heading * 4096)));
        add(static_cast<std::uint32_t>(std::lround(s.speed * 4096)));
        add(static_cast<std::uint32_t>(s.crew + s.aliveCells * 97 + (s.out ? 1 : 0)));
        for (const auto &c : s.cells)
            add(static_cast<std::uint32_t>(std::lround(c.hp * 4096)) ^ (c.alive ? 0u : 0x5f5fu));
    }
    add(static_cast<std::uint32_t>(b.projectiles().size()));
    return h;
}
} // namespace broadside::sim
