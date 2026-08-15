#pragma once
#include "shipyard.h"
#include "sim/rng.h"
#include <string_view>
namespace broadside::game {
struct BuildProfile {
    sim::PartId gun = sim::PartId::GunDeck;
    int gunCount = 4, mastCount = 2;
    sim::PartId armour = sim::PartId::Heavy;
    bool massed = false;
    bool hasSecond = false;
    sim::PartId second = sim::PartId::Swivel;
};
void autoBuild(Shipyard &yard, sim::Rng &rng);
int autoBuild(sim::Design &design, int hullIndex, int budget, const BuildProfile &profile, bool port = true);
const BuildProfile &archetype(std::string_view name);
sim::Design archetypeDesign(std::string_view name, int hullIndex, int budget);
int cumulativeBudget(int hullIndex);
} // namespace broadside::game
