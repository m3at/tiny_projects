#pragma once

#include "geometry.h"
#include "rng.h"
#include "types.h"
#include <cstddef>
#include <string>

namespace broadside::sim {

struct BattleRules {
    double retargetSeconds = 0.6;
    double targetSwitchMargin = 0.8;
    double drawMargin = 0.01;
    double battleCap = BattleCap;
};

struct BattleInit {
    std::vector<Design> designs;
    int hullIndex = 0;
    std::uint32_t seed = 0;
    double windTo = 0.0;
    BattleRules rules{};
};

struct RuntimeShip {
    int index = 0;
    int hullIndex = 0;
    Design design;
    std::vector<RuntimeCell> cells;
    std::vector<Gun> guns;
    Vec2 position{};
    double heading = 0.0;
    double speed = 0.0;
    double sinHeading = 0.0;
    double cosHeading = 1.0;
    double profileRange = 34.0;
    double profileArcBias = 90.0;
    double mass = 1.0;
    double sail = 0.22;
    int sailWanted = 2;
    int crew = 0;
    double crewLost = 0.0;
    int crewSupply = 0;
    int masts = 0;
    int magazines = 0;
    int aliveCells = 0;
    int initialCells = 0;
    double initialStructure = 0.0;
    double hitRadiusSq = 0.0;
    double semiLong = 0.0;
    double semiWide = 0.0;
    bool canFire = false;
    bool out = false;
    double outAt = 0.0;
    int target = -1;
    Ammo ammo = Ammo::Round;
};

class Battle {
  public:
    Battle(std::vector<Design> designs, int hullIndex, std::uint32_t seed, double windTo);
    explicit Battle(BattleInit init);

    void advance(double dt);
    void advanceTicks(int ticks);
    void setAmmo(int seat, Ammo ammo);
    void finish();

    bool over() const {
        return over_;
    }
    int tickCount() const {
        return tickCount_;
    }
    double time() const {
        return time_;
    }
    int hullIndex() const {
        return hullIndex_;
    }
    int shipCount() const {
        return static_cast<int>(ships.size());
    }
    int winner() const {
        return result_.winner;
    }
    const Result &result() const {
        return result_;
    }
    std::uint32_t seed() const {
        return seed_;
    }
    double windTo() const {
        return windTo_;
    }
    std::vector<RuntimeShip> &state() {
        return ships;
    }
    const std::vector<RuntimeShip> &state() const {
        return ships;
    }
    std::vector<Projectile> &projectiles() {
        return projectiles_;
    }
    const std::vector<Projectile> &projectiles() const {
        return projectiles_;
    }
    std::vector<Effect> &effects() {
        return effects_;
    }
    const std::vector<Effect> &effects() const {
        return effects_;
    }
    const std::vector<BattleLog> &log() const {
        return log_;
    }
    double structureFraction(const RuntimeShip &ship) const;

  private:
    void tick();
    void refresh(RuntimeShip &ship);
    void steer(RuntimeShip &ship, RuntimeShip &enemy);
    void fire(RuntimeShip &ship);
    void stepProjectiles();
    void resolveHit(RuntimeShip &ship, RuntimeCell &cell, const Projectile &projectile);
    bool damageCell(RuntimeShip &ship, RuntimeCell &cell, double amount, bool pierce);
    void sever(RuntimeShip &ship);
    void checkEnd();
    RuntimeCell *cellAt(RuntimeShip &ship, Vec2 relative);
    RuntimeCell *nearestCell(RuntimeShip &ship, Vec2 world);
    bool bears(const RuntimeShip &ship, const Gun &gun, Vec2 delta, double distanceSq) const;
    static double hullDamage(int index);

    std::vector<RuntimeShip> ships;
    std::vector<Projectile> projectiles_;
    std::vector<Effect> effects_;
    std::vector<BattleLog> log_;
    Rng rng_;
    std::uint32_t seed_ = 0;
    int hullIndex_ = 0;
    int tickCount_ = 0;
    double time_ = 0.0;
    double carry_ = 0.0;
    double windTo_ = 0.0;
    double windCos_ = 1.0;
    double windSin_ = 0.0;
    int afloat_ = 0;
    int sense_ = 1;
    double arenaRadius_ = 60.0;
    double damageScale_ = 1.0;
    std::vector<double> contactAt_;
    double retargetAt_ = 0.6;
    BattleRules rules_{};
    Result result_;
    bool over_ = false;
};

} // namespace broadside::sim
