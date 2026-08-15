#pragma once
#include "sim/battle.h"
#include "sim/rng.h"
namespace broadside::game {
sim::Ammo chooseAmmo(const sim::RuntimeShip &me, const sim::RuntimeShip &enemy);
class AmmoBot {
  public:
    AmmoBot(sim::Battle &b, std::vector<int> seats, sim::Rng) : battle_(b), seats_(std::move(seats)) {}
    void update(float dt);

  private:
    sim::Battle &battle_;
    std::vector<int> seats_;
    float next_ = 0;
};
} // namespace broadside::game
