#include "match.h"
#include "sim/geometry.h"
#include "sim/rng.h"
#include <algorithm>
#include <cmath>
namespace broadside::game {
const std::vector<RoundInfo> &rounds() {
    static const std::vector<RoundInfo> r = {{0, 34, 40}, {1, 32, 26}, {2, 42, 26}, {3, 46, 28}, {4, 56, 30}};
    return r;
}
Match makeMatch(std::uint32_t seed, int players) {
    Match m;
    m.seed = seed;
    m.players = players;
    m.scores.assign(players, 0);
    m.scrap.assign(players, 0);
    m.designs.resize(players, sim::makeDesign());
    m.damage.assign(players, 0);
    return m;
}
void beginRound(Match &m) {
    auto r = rounds()[m.round];
    sim::Rng rng(sim::hashSeed({m.seed, static_cast<std::uint32_t>(m.round), 0x9e3779b9u}));
    m.wind = rng.range(0, sim::Pi * 2);
    for (auto &s : m.scrap)
        s += r.scrap;
    for (auto &d : m.designs)
        sim::fitDesignToHull(d, r.hull);
}
void recordResult(Match &m, const sim::Result &r) {
    if (r.winner >= 0 && r.winner < m.players)
        m.scores[r.winner]++;
    for (int place = 1; place < static_cast<int>(r.placing.size()); ++place)
        m.scrap[r.placing[place]] += static_cast<int>(
            std::lround(rounds()[m.round].scrap * .45f * static_cast<float>(place) / (m.players - 1)));
    if (winner(m) >= 0) {
        m.over = true;
        return;
    }
    if (++m.round >= static_cast<int>(rounds().size()))
        m.over = true;
}
int winner(const Match &m) {
    for (int i = 0; i < m.players; ++i)
        if (m.scores[i] >= 3)
            return i;
    if (!m.over && m.round < static_cast<int>(rounds().size()))
        return -1;
    int leader = -1, best = -1;
    bool tied = false;
    for (int i = 0; i < m.players; ++i) {
        if (m.scores[i] > best) {
            best = m.scores[i];
            leader = i;
            tied = false;
        } else if (m.scores[i] == best)
            tied = true;
    }
    return tied ? -1 : leader;
}
} // namespace broadside::game
