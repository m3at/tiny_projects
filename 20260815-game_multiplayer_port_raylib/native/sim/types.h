#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace broadside::sim {

struct Vec2 {
    double x = 0.0;
    double z = 0.0;
};
struct Coord {
    int dx = 0;
    int dz = 0;
    friend bool operator<(const Coord &a, const Coord &b) {
        return a.dz < b.dz || (a.dz == b.dz && a.dx < b.dx);
    }
    friend bool operator==(const Coord &a, const Coord &b) {
        return a.dx == b.dx && a.dz == b.dz;
    }
};

enum class Ammo : std::uint8_t { Round, Grape };
enum class PartId : std::uint8_t {
    Timber,
    Heavy,
    Crew,
    Mast,
    Magazine,
    Swivel,
    GunDeck,
    Carronade,
    LongGun,
    Helm
};
enum class Arc : std::uint8_t { All, Side, Bow };

struct GunSpec {
    Arc arc = Arc::All;
    double halfArc = 180.0;
    double range = 0.0;
    double reload = 1.0;
    int shots = 1;
    double spread = 0.0;
    double speed = 1.0;
    double roundDamage = 0.0;
    double grapeDamage = 0.0;
    double grapeCrew = 0.0;
    bool pierce = false;
};

struct PartDef {
    PartId id;
    const char *name;
    char glyph;
    int cost;
    double hp;
    int crewSupply = 0;
    int crewCost = 0;
    int soak = 0;
    bool magazine = false;
    bool fixed = false;
    bool bowOnly = false;
    bool gun = false;
    GunSpec gunSpec{};
};

struct HullDef {
    const char *name;
    int width = 0;
    int length = 0;
    int bowZ = 0;
    int bowLimit = 0;
    std::vector<Coord> cells;
};

struct Slot {
    PartId id = PartId::Timber;
    double hp = 0.0;
};
struct Design {
    std::map<Coord, Slot> parts;
    std::vector<Coord> order;
};

struct RuntimeCell {
    Coord coord{};
    PartId id = PartId::Timber;
    double hp = 0.0;
    double maxHp = 0.0;
    bool alive = true;
    int crewCost = 0;
    int crewSupply = 0;
    int soak = 0;
    bool magazine = false;
    bool gun = false;
    std::uint32_t reached = 0;
};

struct Gun {
    int cell = -1;
    GunSpec spec{};
    double arc = 0.0;
    double readyAt = 0.0;
    bool manned = false;
};

struct Projectile {
    Vec2 pos{};
    Vec2 velocity{};
    int owner = 0;
    int target = 0;
    double damage = 0.0;
    double crew = 0.0;
    double ttl = 0.0;
    bool pierce = false;
    Ammo kind = Ammo::Round;
};

struct Effect {
    enum class Type { Muzzle, Impact, Splash, Destroy, Crew, Sever, Detonate, Ammo };
    Type type;
    Vec2 pos{};
    int ship = -1;
    PartId part = PartId::Timber;
    Ammo ammo = Ammo::Round;
    double heading = 0.0;
    int weaponWeight = 0;
};
struct BattleLog {
    double time = 0.0;
    int ship = -1;
    std::string text;
};
struct Result {
    int winner = -1;
    std::vector<int> placing;
    std::string reason;
};

constexpr double CellSize = 2.4;
constexpr double TickSeconds = 1.0 / 60.0;
constexpr double BattleCap = 40.0;
constexpr int MaxPlayers = 4;

const PartDef &part(PartId id);
const std::vector<PartId> &buyableParts();
const HullDef &hull(int index);
const std::vector<HullDef> &hulls();
Design makeDesign();
Design cloneDesign(const Design &design);
void setPart(Design &design, Coord coord, Slot slot);
void erasePart(Design &design, Coord coord);
void fitDesignToHull(Design &design, int hullIndex);
Coord coord(int dx, int dz);
bool isHullCell(int hullIndex, Coord c);
bool isBowCell(int hullIndex, int dz);
int hullCellCount(int hullIndex);

} // namespace broadside::sim
