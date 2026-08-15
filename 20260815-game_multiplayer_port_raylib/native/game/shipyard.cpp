#include "shipyard.h"
#include <algorithm>
#include <cmath>

namespace broadside::game {
using namespace sim;
Shipyard::Shipyard(Design design, int hull, int scrap)
    : design_(std::move(design)), hull_(hull), scrap_(scrap) {
    fitDesignToHull(design_, hull_);
    recalculate();
}
ActionResult Shipyard::place(Coord c, PartId id) {
    const auto &pd = part(id);
    if (!isHullCell(hull_, c))
        return {false, "Not part of the hull."};
    if (design_.parts.contains(c))
        return {false, "Cell already occupied."};
    if (!offer_.empty() && std::find(offer_.begin(), offer_.end(), id) == offer_.end())
        return {false, "That part is not in the current offer."};
    if (scrap_ < pd.cost)
        return {false, "Not enough scrap."};
    if (pd.gunSpec.arc == Arc::Side && c.dx == 0)
        return {false, "Broadside guns need a flank cell."};
    if (pd.gunSpec.arc == Arc::Bow && !isBowCell(hull_, c.dz))
        return {false, "A bow chaser has to be worked from the bow."};
    setPart(design_, c, {id, pd.hp});
    scrap_ -= pd.cost;
    refundable_[c] = pd.cost;
    recalculate();
    return {true, {}};
}
ActionResult Shipyard::remove(Coord c) {
    auto it = design_.parts.find(c);
    if (it == design_.parts.end() || it->second.id == PartId::Helm)
        return {false, "The helm is fixed."};
    if (auto fresh = refundable_.find(c); fresh != refundable_.end()) {
        scrap_ += fresh->second;
        refundable_.erase(fresh);
    }
    erasePart(design_, c);
    recalculate();
    return {true, {}};
}
ActionResult Shipyard::refit() {
    struct Damage {
        Coord coord;
        double fraction;
        int cost;
    };
    std::vector<Damage> damaged;
    for (const auto &[c, s] : design_.parts) {
        const auto &p = part(s.id);
        if (s.hp < p.hp)
            damaged.push_back({c, s.hp / p.hp, std::max(1, static_cast<int>(std::ceil(p.cost * .5f)))});
    }
    std::sort(damaged.begin(), damaged.end(), [](const Damage &a, const Damage &b) {
        return a.fraction != b.fraction ? a.fraction < b.fraction : a.coord < b.coord;
    });
    int repaired = 0;
    for (const auto &d : damaged) {
        if (d.cost > scrap_)
            continue;
        scrap_ -= d.cost;
        design_.parts[d.coord].hp = part(design_.parts[d.coord].id).hp;
        ++repaired;
    }
    if (repaired == 0)
        return {false, damaged.empty() ? "Nothing needs refitting." : "Not enough scrap to refit."};
    recalculate();
    return {true, {}};
}
void Shipyard::makeOffer(Rng &rng) {
    offer_ = {PartId::Timber, PartId::Magazine, PartId::Crew};
    const auto &parts = buyableParts();
    while (offer_.size() < 5) {
        PartId id = parts[rng.integer(0, static_cast<int>(parts.size() - 1))];
        if (std::find(offer_.begin(), offer_.end(), id) == offer_.end())
            offer_.push_back(id);
    }
}
ActionResult Shipyard::reroll(Rng &rng) {
    if (scrap_ < 2)
        return {false, "A reroll costs two scrap."};
    scrap_ -= 2;
    makeOffer(rng);
    return {true, {}};
}
void Shipyard::recalculate() {
    stats_ = {};
    stats_.total = hullCellCount(hull_);
    stats_.used = static_cast<int>(design_.parts.size());
    for (const auto &[c, s] : design_.parts) {
        const auto &p = part(s.id);
        stats_.crewSupply += p.crewSupply;
        stats_.crewNeeded += p.crewCost;
        stats_.magazines += p.magazine;
        stats_.masts += s.id == PartId::Mast;
        stats_.guns += p.gun;
    }
    stats_.mastsWanted = std::max(2, (stats_.total + 9) / 10);
    int pool = stats_.crewSupply;
    for (const auto &[c, s] : design_.parts) {
        const auto &p = part(s.id);
        if (!p.crewCost)
            continue;
        if (pool >= p.crewCost)
            pool -= p.crewCost;
        else
            stats_.unmanned.push_back(c);
    }
    if (stats_.guns == 0)
        stats_.warnings.push_back("No guns.");
    else if (stats_.magazines == 0)
        stats_.warnings.push_back("No powder magazine.");
    if (!stats_.unmanned.empty())
        stats_.warnings.push_back("Some stations are unmanned.");
    if (stats_.masts == 0)
        stats_.warnings.push_back("No masts: the ship will barely move.");
    if (stats_.masts > stats_.mastsWanted)
        stats_.warnings.push_back("This hull cannot use all those masts.");
    if (stats_.total - stats_.used > stats_.total * .3f)
        stats_.warnings.push_back("Open holes let shot reach the spine.");
}
} // namespace broadside::game
