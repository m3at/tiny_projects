#include "autobuild.h"
#include "match.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

namespace broadside::game {
using namespace sim;
namespace {
std::vector<Coord> freeCells(const Design &d, int hullIndex) {
    std::vector<Coord> out;
    for (auto c : hull(hullIndex).cells)
        if (!d.parts.contains(c))
            out.push_back(c);
    return out;
}
std::vector<Coord> spine(std::vector<Coord> cells) {
    cells.erase(std::remove_if(cells.begin(), cells.end(), [](Coord c) { return c.dx != 0; }), cells.end());
    std::stable_sort(cells.begin(), cells.end(), [](Coord a, Coord b) {
        return std::abs(a.dz) != std::abs(b.dz) ? std::abs(a.dz) < std::abs(b.dz) : a.dz < b.dz;
    });
    return cells;
}
std::vector<Coord> flanks(std::vector<Coord> cells, int side) {
    std::vector<Coord> port, star;
    for (auto c : cells) {
        if (c.dx < 0)
            port.push_back(c);
        else if (c.dx > 0)
            star.push_back(c);
    }
    auto byMid = [](Coord a, Coord b) { return std::abs(a.dz) < std::abs(b.dz); };
    std::stable_sort(port.begin(), port.end(), byMid);
    std::stable_sort(star.begin(), star.end(), byMid);
    if (side < 0) {
        port.insert(port.end(), star.begin(), star.end());
        return port;
    }
    if (side > 0) {
        star.insert(star.end(), port.begin(), port.end());
        return star;
    }
    std::vector<Coord> out;
    for (std::size_t i = 0; i < std::max(port.size(), star.size()); ++i) {
        if (i < port.size())
            out.push_back(port[i]);
        if (i < star.size())
            out.push_back(star[i]);
    }
    return out;
}
void place(Design &d, Coord c, PartId id) {
    setPart(d, c, {id, part(id).hp});
}
std::vector<Coord> gunCells(const Design &d, int hullIndex, PartId id, int massedSide) {
    auto free = freeCells(d, hullIndex);
    const auto arc = part(id).gunSpec.arc;
    if (arc == Arc::Side) {
        auto out = flanks(free, massedSide);
        out.erase(std::remove_if(out.begin(), out.end(), [](Coord c) { return c.dx == 0; }), out.end());
        return out;
    }
    auto out = flanks(free, 0);
    auto middle = spine(free);
    out.insert(out.end(), middle.begin(), middle.end());
    if (arc == Arc::Bow)
        out.erase(
            std::remove_if(out.begin(), out.end(), [&](Coord c) { return !isBowCell(hullIndex, c.dz); }),
            out.end());
    return out;
}
} // namespace

int autoBuild(Design &design, int hullIndex, int budget, const BuildProfile &profile, bool portSide) {
    const PartId gunId = profile.gun;
    const int gunCrew = part(gunId).crewCost;
    int wantGuns = profile.gunCount, wantMasts = profile.mastCount, scrap = budget;
    if (!design.parts.contains({0, 0}))
        place(design, {0, 0}, PartId::Helm);
    auto crewFor = [&](int guns, int masts) {
        return static_cast<int>(std::ceil((guns * gunCrew + masts) / 3.0));
    };
    auto costOf = [&](int guns, int masts) {
        return part(gunId).cost * guns + part(PartId::Mast).cost * masts +
               part(PartId::Crew).cost * crewFor(guns, masts) + part(PartId::Magazine).cost;
    };
    while (wantGuns > 0 && costOf(wantGuns, wantMasts) > scrap) {
        if (wantMasts > 1)
            --wantMasts;
        else
            --wantGuns;
    }
    if (wantGuns == 0)
        wantMasts = std::max(0, std::min(wantMasts, (scrap - 4) / part(PartId::Mast).cost));
    const auto &hd = hull(hullIndex);
    const auto arc = part(gunId).gunSpec.arc;
    int flankRoom = 0;
    if (arc == Arc::Bow)
        flankRoom = static_cast<int>(std::count_if(hd.cells.begin(), hd.cells.end(),
                                                   [&](Coord c) { return isBowCell(hullIndex, c.dz); }));
    else {
        flankRoom = static_cast<int>(
            std::count_if(hd.cells.begin(), hd.cells.end(), [](Coord c) { return c.dx != 0; }));
        if (arc != Arc::Side)
            flankRoom += std::max(0, static_cast<int>(std::count_if(hd.cells.begin(), hd.cells.end(),
                                                                    [](Coord c) { return c.dx == 0; })) -
                                         4);
    }
    int spineRoom =
        static_cast<int>(std::count_if(hd.cells.begin(), hd.cells.end(), [](Coord c) { return c.dx == 0; })) -
        1;
    auto holeReserve = [&](int guns, int masts) {
        return static_cast<int>(hd.cells.size()) - (2 + guns + masts + crewFor(guns, masts));
    };
    auto affordable = [&](int guns, int masts) {
        return costOf(guns, masts) + std::max(0, holeReserve(guns, masts)) <= scrap;
    };
    auto roomFor = [&](int guns, int masts) {
        return 2 + guns + masts + crewFor(guns, masts) <= static_cast<int>(hd.cells.size());
    };
    while (wantGuns + 1 <= flankRoom - 2 && affordable(wantGuns + 1, wantMasts) &&
           roomFor(wantGuns + 1, wantMasts) && wantGuns < 24)
        ++wantGuns;
    while (wantMasts + 1 <= spineRoom - 2 && affordable(wantGuns, wantMasts + 1) &&
           roomFor(wantGuns, wantMasts + 1) && wantMasts < 5)
        ++wantMasts;
    auto spend = [&](PartId id) {
        if (scrap < part(id).cost)
            return false;
        scrap -= part(id).cost;
        return true;
    };
    {
        auto cells = spine(freeCells(design, hullIndex));
        if (!cells.empty()) {
            auto spot = cells[cells.size() > 2 ? 1 : 0];
            if (spend(PartId::Magazine))
                place(design, spot, PartId::Magazine);
        }
    }
    {
        bool wantsBow = part(gunId).gunSpec.arc == Arc::Bow ||
                        (profile.hasSecond && part(profile.second).gunSpec.arc == Arc::Bow);
        auto cells = spine(freeCells(design, hullIndex));
        cells.erase(std::remove_if(cells.begin(), cells.end(),
                                   [&](Coord c) { return wantsBow && isBowCell(hullIndex, c.dz); }),
                    cells.end());
        std::sort(cells.begin(), cells.end(), [](Coord a, Coord b) { return a.dz < b.dz; });
        for (int i = 0; i < wantMasts && i < static_cast<int>(cells.size()); ++i)
            if (spend(PartId::Mast))
                place(design, cells[i], PartId::Mast);
    }
    {
        auto cells = gunCells(design, hullIndex, gunId, profile.massed ? (portSide ? -1 : 1) : 0);
        for (int i = 0; i < wantGuns && i < static_cast<int>(cells.size()); ++i)
            if (spend(gunId))
                place(design, cells[i], gunId);
    }
    if (profile.hasSecond) {
        const PartId second = profile.second;
        for (auto cell : gunCells(design, hullIndex, second, profile.massed ? (portSide ? -1 : 1) : 0)) {
            int hands = part(second).crewCost;
            for (const auto &[_, slot] : design.parts)
                hands += part(slot.id).crewCost;
            int quarters = static_cast<int>(std::ceil(hands / 3.0));
            int cellsLeft = static_cast<int>(freeCells(design, hullIndex).size()) - 1;
            int plugs = std::max(0, cellsLeft - quarters);
            if (quarters > cellsLeft ||
                part(second).cost + quarters * part(PartId::Crew).cost + plugs > scrap || !spend(second))
                break;
            place(design, cell, second);
        }
    }
    {
        int need = 0;
        for (const auto &[_, slot] : design.parts)
            need += part(slot.id).crewCost;
        int quarters = static_cast<int>(std::ceil(need / 3.0));
        auto cells = spine(freeCells(design, hullIndex));
        std::sort(cells.begin(), cells.end(), [](Coord a, Coord b) { return a.dz > b.dz; });
        auto sides = flanks(freeCells(design, hullIndex), 0);
        std::reverse(sides.begin(), sides.end());
        cells.insert(cells.end(), sides.begin(), sides.end());
        for (int i = 0; i < quarters && i < static_cast<int>(cells.size()); ++i)
            if (spend(PartId::Crew))
                place(design, cells[i], PartId::Crew);
    }
    {
        auto cells = freeCells(design, hullIndex);
        std::sort(cells.begin(), cells.end(), [](Coord a, Coord b) {
            return std::abs(a.dx) != std::abs(b.dx) ? std::abs(a.dx) > std::abs(b.dx)
                                                    : std::abs(a.dz) < std::abs(b.dz);
        });
        int holes = static_cast<int>(cells.size());
        for (auto c : cells) {
            int upgrade = part(profile.armour).cost - part(PartId::Timber).cost;
            if (profile.armour != PartId::Timber && scrap - holes >= upgrade &&
                scrap >= part(profile.armour).cost) {
                scrap -= part(profile.armour).cost;
                place(design, c, profile.armour);
            } else if (scrap >= part(PartId::Timber).cost) {
                scrap -= part(PartId::Timber).cost;
                place(design, c, PartId::Timber);
            } else
                break;
            --holes;
        }
    }
    return scrap;
}

const BuildProfile &archetype(std::string_view name) {
    static const std::map<std::string_view, BuildProfile> profiles = {
        {"brawler", {PartId::GunDeck, 4, 2, PartId::Heavy, false, false, PartId::Swivel}},
        {"massed", {PartId::GunDeck, 4, 2, PartId::Heavy, true, false, PartId::Swivel}},
        {"sniper", {PartId::LongGun, 4, 3, PartId::Timber, false, true, PartId::GunDeck}},
        {"harasser", {PartId::Swivel, 6, 4, PartId::Timber, false, false, PartId::Swivel}},
        {"crusher", {PartId::Carronade, 4, 3, PartId::Heavy, false, false, PartId::Swivel}},
        {"mixed", {PartId::GunDeck, 3, 3, PartId::Heavy, false, true, PartId::Swivel}}};
    auto it = profiles.find(name);
    if (it == profiles.end())
        throw std::invalid_argument("unknown archetype");
    return it->second;
}
Design archetypeDesign(std::string_view name, int hullIndex, int budget) {
    Design design = makeDesign();
    autoBuild(design, hullIndex, budget, archetype(name));
    return design;
}
int cumulativeBudget(int hullIndex) {
    int total = 0;
    for (int i = 0; i <= hullIndex; ++i)
        total += rounds()[i].scrap;
    return total;
}

void autoBuild(Shipyard &yard, Rng &rng) {
    BuildProfile profile;
    const PartId guns[] = {PartId::Swivel, PartId::GunDeck, PartId::Carronade};
    profile.gun = guns[rng.integer(0, 2)];
    profile.gunCount = rng.integer(1, 8);
    profile.mastCount = rng.integer(1, std::max(2, (hullCellCount(yard.hullIndex()) + 9) / 10) + 1);
    profile.armour = rng.chance(.5f) ? PartId::Heavy : PartId::Timber;
    profile.massed = rng.chance(.35f);
    Design design = yard.design();
    int left = autoBuild(design, yard.hullIndex(), yard.scrap(), profile);
    yard.adopt(std::move(design), left);
}
} // namespace broadside::game
