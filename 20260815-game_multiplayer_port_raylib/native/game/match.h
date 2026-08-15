#pragma once
#include "sim/types.h"
#include <cstdint>
#include <vector>
namespace broadside::game {
struct RoundInfo {
    int hull = 0, scrap = 0, buildSeconds = 0;
};
struct Match {
    std::uint32_t seed = 0;
    int round = 0, players = 2;
    std::vector<int> scores, scrap;
    std::vector<sim::Design> designs;
    std::vector<float> damage;
    float wind = 0;
    bool over = false;
};
const std::vector<RoundInfo> &rounds();
Match makeMatch(std::uint32_t seed, int players);
void beginRound(Match &m);
void recordResult(Match &m, const sim::Result &r);
int winner(const Match &m);
} // namespace broadside::game
