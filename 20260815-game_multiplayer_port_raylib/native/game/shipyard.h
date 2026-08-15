#pragma once
#include "sim/rng.h"
#include "sim/types.h"
#include <map>
#include <string>
#include <vector>

namespace broadside::game {
struct Stats {
    int used = 0, total = 0, crewSupply = 0, crewNeeded = 0, magazines = 0, masts = 0, mastsWanted = 0,
        guns = 0;
    std::vector<sim::Coord> unmanned;
    std::vector<std::string> warnings;
};
struct ActionResult {
    bool ok = false;
    std::string why;
};
class Shipyard {
  public:
    Shipyard(sim::Design design, int hull, int scrap);
    ActionResult place(sim::Coord c, sim::PartId id);
    ActionResult remove(sim::Coord c);
    ActionResult refit();
    ActionResult reroll(sim::Rng &rng);
    void setOffer(std::vector<sim::PartId> offer) {
        offer_ = std::move(offer);
    }
    const sim::Design &design() const {
        return design_;
    }
    sim::Design &design() {
        return design_;
    }
    int scrap() const {
        return scrap_;
    }
    int hullIndex() const {
        return hull_;
    }
    const Stats &stats() const {
        return stats_;
    }
    const std::vector<sim::PartId> &offer() const {
        return offer_;
    }
    void adopt(sim::Design design, int scrap) {
        design_ = std::move(design);
        scrap_ = scrap;
        refundable_.clear();
        recalculate();
    }
    void recalculate();

  private:
    void makeOffer(sim::Rng &rng);
    sim::Design design_;
    int hull_;
    int scrap_;
    std::vector<sim::PartId> offer_;
    std::map<sim::Coord, int> refundable_;
    Stats stats_;
};
} // namespace broadside::game
