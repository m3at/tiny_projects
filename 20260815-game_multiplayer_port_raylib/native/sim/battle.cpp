#include "battle.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace broadside::sim {
namespace {
constexpr double Arena = 60.0, BaseSpeed = 13.5, BaseTurn = 1.15, WindMin = 0.35;
constexpr double MinSeparation = 9.0, OrbitTolerance = 0.6, OrbitClose = 0.55;
constexpr double Tick = TickSeconds, OvertimeAt = 20.0, OvertimeRamp = 0.14, OvertimeMax = 1.6;
double overtime(double time) {
    return 1.0 + std::min(OvertimeMax, std::max(0.0, (time - OvertimeAt) * OvertimeRamp));
}
Vec2 direction(double heading) {
    return {fsin(heading), -fcos(heading)};
}
double sq(Vec2 v) {
    return v.x * v.x + v.z * v.z;
}
double reachAlong(const RuntimeShip &s, double nx, double nz) {
    const double fwd = nx * s.sinHeading - nz * s.cosHeading;
    const double side = nx * s.cosHeading + nz * s.sinHeading;
    const double bf = s.semiWide * fwd, as = s.semiLong * side;
    return (s.semiLong * s.semiWide) / std::sqrt(bf * bf + as * as);
}
} // namespace

double Battle::hullDamage(int index) {
    static constexpr double values[] = {1.05, 0.5, 0.36, 0.24, 0.2};
    return values[index < 5 ? index : 4];
}

Battle::Battle(std::vector<Design> designs, int hullIndex, std::uint32_t seed, double windTo)
    : Battle(BattleInit{std::move(designs), hullIndex, seed, windTo, {}}) {}

Battle::Battle(BattleInit init)
    : rng_(init.seed), seed_(init.seed), hullIndex_(init.hullIndex), windTo_(init.windTo),
      windCos_(fcos(init.windTo)), windSin_(fsin(init.windTo)), rules_(init.rules) {
    auto designs = std::move(init.designs);
    const int hullIndex = hullIndex_;
    const int count = static_cast<int>(designs.size());
    if (count < 2 || count > MaxPlayers)
        throw std::invalid_argument("Broadside needs two to four ships");
    afloat_ = count;
    sense_ = rng_.chance(0.5) ? 1 : -1;
    arenaRadius_ = Arena * (count == 4 ? 1.32 : count == 3 ? 1.18 : 1.0);
    damageScale_ = count == 4 ? 3.0 : count == 3 ? 2.0 : 1.0;
    contactAt_.assign(count * count, 0.0);
    retargetAt_ = rules_.retargetSeconds;
    const double radius = arenaRadius_ * .52;
    for (int i = 0; i < count; ++i) {
        RuntimeShip s;
        s.index = i;
        s.hullIndex = hullIndex;
        s.design = cloneDesign(designs[i]);
        const auto &hd = hull(hullIndex);
        s.sailWanted = std::max(2, (static_cast<int>(hd.cells.size()) + 9) / 10);
        if (count == 2) {
            s.position = {i == 0 ? -9.0f : 9.0f, i == 0 ? 24.0f : -24.0f};
            s.heading = i == 0 ? 0.0f : Pi;
        } else {
            const double theta = 2 * Pi * static_cast<double>(i) / count;
            s.position = {radius * fsin(theta), -radius * fcos(theta)};
            s.heading = fatan2(-s.position.x, s.position.z);
        }
        s.sinHeading = fsin(s.heading);
        s.cosHeading = fcos(s.heading);
        std::vector<Coord> ordered = s.design.order;
        for (const auto &[c, _] : s.design.parts)
            if (std::find(ordered.begin(), ordered.end(), c) == ordered.end())
                ordered.push_back(c);
        for (const auto &c : ordered) {
            const auto &slot = s.design.parts.at(c);
            const auto &pd = part(slot.id);
            RuntimeCell cell;
            cell.coord = c;
            cell.id = slot.id;
            cell.hp = slot.hp;
            cell.maxHp = pd.hp;
            cell.crewCost = pd.crewCost;
            cell.crewSupply = pd.crewSupply;
            cell.soak = pd.soak;
            cell.magazine = pd.magazine;
            cell.gun = pd.gun;
            s.cells.push_back(cell);
        }
        for (int c = 0; c < static_cast<int>(s.cells.size()); ++c)
            if (part(s.cells[c].id).gun) {
                const auto &pd = part(s.cells[c].id);
                Gun g;
                g.cell = c;
                g.spec = pd.gunSpec;
                g.arc = pd.gunSpec.arc == Arc::Side ? (s.cells[c].coord.dx < 0 ? -Pi / 2 : Pi / 2) : 0.0f;
                s.guns.push_back(g);
            }
        auto key = [&](const Gun &gun) {
            const auto c = s.cells[gun.cell].coord;
            return std::to_string(c.dx) + "," + std::to_string(c.dz);
        };
        std::sort(s.guns.begin(), s.guns.end(), [&](const Gun &a, const Gun &b) {
            const bool aa = a.arc == 0 || (sense_ > 0 ? a.arc < 0 : a.arc > 0);
            const bool bb = b.arc == 0 || (sense_ > 0 ? b.arc < 0 : b.arc > 0);
            if (aa != bb)
                return aa > bb;
            return key(a) < key(b);
        });
        s.aliveCells = static_cast<int>(s.cells.size());
        s.initialCells = s.aliveCells;
        for (const auto &c : s.cells)
            s.initialStructure += c.maxHp;
        double farSq = 0;
        for (const auto &c : s.cells) {
            double lx = c.coord.dx * CellSize, lz = c.coord.dz * CellSize;
            farSq = std::max(farSq, lx * lx + lz * lz);
        }
        double hit = std::sqrt(farSq) + CellSize * std::sqrt(.5) + 1e-9;
        s.hitRadiusSq = hit * hit;
        s.semiLong = hull(hullIndex).length * .5 * CellSize;
        s.semiWide = hull(hullIndex).width * .5 * CellSize;
        s.target = (i + 1) % count;
        refresh(s);
        for (auto &g : s.guns)
            g.readyAt = rng_.range(0, g.spec.reload * .35f);
        ships.push_back(std::move(s));
    }
    for (auto &s : ships) {
        double w = 0, r = 0, b = 0;
        for (const auto &g : s.guns) {
            const auto &sp = g.spec;
            double weight = sp.shots * sp.roundDamage / sp.reload;
            w += weight;
            r += weight * sp.range;
            b += weight * (sp.arc == Arc::Bow ? 0 : 90);
        }
        if (w > 0) {
            s.profileRange = std::clamp(r / w, 16.0, 88.0) * .85;
            s.profileArcBias = b / w;
        }
    }
}

void Battle::refresh(RuntimeShip &s) {
    int supply = 0, masts = 0, mags = 0;
    for (const auto &c : s.cells)
        if (c.alive) {
            supply += c.crewSupply;
            masts += c.id == PartId::Mast;
            mags += c.magazine;
        }
    s.crewSupply = supply;
    s.crew = std::max(0, static_cast<int>(std::floor(supply - s.crewLost)));
    s.masts = masts;
    s.magazines = mags;
    s.mass = 12.0 / (12.0 + std::max(1, s.aliveCells) * .42);
    s.sail = .22 + .78 * std::min(1.0, static_cast<double>(masts) / s.sailWanted);
    int sailHands = 0;
    for (const auto &c : s.cells)
        if (c.alive && !c.gun)
            sailHands += c.crewCost;
    int pool = std::max(0, s.crew - sailHands);
    bool any = false;
    for (auto &g : s.guns) {
        auto &c = s.cells[g.cell];
        g.manned = c.alive && pool >= c.crewCost;
        if (g.manned) {
            pool -= c.crewCost;
            any = true;
        }
    }
    s.canFire = s.magazines > 0 && any;
}

void Battle::advance(double dt) {
    carry_ += std::min(dt, .25);
    const int n = static_cast<int>(std::floor(carry_ / Tick));
    carry_ -= n * Tick;
    advanceTicks(n);
}
void Battle::advanceTicks(int ticks) {
    for (int i = 0; i < ticks && !over_; ++i) {
        tick();
        ++tickCount_;
    }
}

void Battle::setAmmo(int seat, Ammo ammo) {
    if (over_ || seat < 0 || seat >= static_cast<int>(ships.size()) || ships[seat].ammo == ammo)
        return;
    ships[seat].ammo = ammo;
    for (auto &g : ships[seat].guns)
        g.readyAt = std::max(g.readyAt, time_ + 1.3);
    effects_.push_back(
        {Effect::Type::Ammo, ships[seat].position, seat, PartId::Timber, ammo, ships[seat].heading, 0});
    log_.push_back({time_, seat, ammo == Ammo::Round ? "round shot loaded" : "grape shot loaded"});
}

void Battle::steer(RuntimeShip &s, RuntimeShip &e) {
    Vec2 delta{e.position.x - s.position.x, e.position.z - s.position.z};
    double d = length(delta), bearing = fatan2(delta.x, -delta.z), R = s.profileRange,
           err = std::clamp((d - R) / (R * OrbitTolerance), -1.0, 1.0),
           hold = s.profileArcBias * Pi / 180.0 * sense_, away = Pi * sense_,
           close = OrbitClose + (1 - OrbitClose) * err,
           alpha = err > 0 ? hold * (1 - err * close) : hold + (away - hold) * (-err * 0.0),
           desired = bearing + alpha;
    if (sq(s.position) > arenaRadius_ * .8 * arenaRadius_ * .8) {
        double radial = length(s.position), inward = fatan2(-s.position.x, s.position.z),
               pull = std::min(1.0, (radial - arenaRadius_ * .8) / (arenaRadius_ * .25));
        desired += wrapAngle(inward - desired) * pull;
    }
    double drive = s.sail * s.mass, turn = BaseTurn * drive, diff = wrapAngle(desired - s.heading);
    s.heading = wrapAngle(s.heading + std::clamp(diff, -turn * Tick, turn * Tick));
    s.cosHeading = fcos(s.heading);
    s.sinHeading = fsin(s.heading);
    double wind = WindMin + (1 - WindMin) * (s.cosHeading * windCos_ + s.sinHeading * windSin_ + 1) / 2,
           target = BaseSpeed * wind * drive;
    s.speed += (target - s.speed) * std::min(1.0, Tick * 1.4);
    s.position.x += s.sinHeading * s.speed * Tick;
    s.position.z -= s.cosHeading * s.speed * Tick;
}

bool Battle::bears(const RuntimeShip &s, const Gun &g, Vec2 d, double distSq) const {
    if (g.spec.arc == Arc::All)
        return true;
    double ac = fcos(g.arc), as = fsin(g.arc), ax = s.sinHeading * ac + s.cosHeading * as,
           az = -s.cosHeading * ac + s.sinHeading * as, dot = d.x * ax + d.z * az;
    if (dot <= 0)
        return false;
    double c = fcos(g.spec.halfArc * Pi / 180.0);
    return dot * dot >= c * c * distSq;
}

void Battle::fire(RuntimeShip &s) {
    if (s.magazines == 0)
        return;
    for (auto &g : s.guns) {
        if (!g.manned || time_ < g.readyAt)
            continue;
        auto &cell = s.cells[g.cell];
        Vec2 muzzle = localToWorld(s.position, s.cosHeading, s.sinHeading, cell.coord);
        RuntimeShip *enemy = nullptr;
        double best = std::numeric_limits<double>::infinity();
        for (auto &e : ships) {
            if (e.index == s.index || e.out)
                continue;
            Vec2 d{e.position.x - muzzle.x, e.position.z - muzzle.z};
            double dsq = sq(d);
            if (dsq > g.spec.range * g.spec.range || dsq >= best || !bears(s, g, d, dsq))
                continue;
            best = dsq;
            enemy = &e;
        }
        if (!enemy)
            continue;
        double flight = std::sqrt(best) / g.spec.speed,
               aimX = enemy->position.x + enemy->sinHeading * enemy->speed * flight,
               aimZ = enemy->position.z - enemy->cosHeading * enemy->speed * flight,
               bearing = fatan2(aimX - muzzle.x, -(aimZ - muzzle.z));
        const bool grape = s.ammo == Ammo::Grape;
        int count = g.spec.shots + (grape ? 1 : 0);
        double spread = g.spec.spread * Pi / 180 * (grape ? 1.8 : 1);
        for (int i = 0; i < count; ++i) {
            double a = bearing + rng_.range(-spread, spread), speed = g.spec.speed * rng_.range(.94, 1.06);
            Projectile p;
            p.pos = muzzle;
            p.velocity = {fsin(a) * speed, -fcos(a) * speed};
            p.owner = s.index;
            p.target = enemy->index;
            p.damage = grape ? g.spec.grapeDamage : g.spec.roundDamage;
            p.crew = grape ? g.spec.grapeCrew * .15 : 0;
            p.ttl = g.spec.range / g.spec.speed * 1.35;
            p.pierce = g.spec.pierce;
            p.kind = s.ammo;
            projectiles_.push_back(p);
        }
        effects_.push_back({Effect::Type::Muzzle, muzzle, s.index, cell.id, s.ammo, s.heading,
                            static_cast<int>(std::round(g.spec.roundDamage))});
        g.readyAt = time_ + g.spec.reload * rng_.range(.65, 1.35);
    }
}

RuntimeCell *Battle::cellAt(RuntimeShip &s, Vec2 r) {
    double x = (r.x * s.cosHeading + r.z * s.sinHeading) / CellSize,
           z = (-r.x * s.sinHeading + r.z * s.cosHeading) / CellSize;
    Coord c{static_cast<int>(std::floor(x + .5)), static_cast<int>(std::floor(z + .5))};
    for (auto &cell : s.cells)
        if (cell.coord == c && cell.alive)
            return &cell;
    return nullptr;
}
RuntimeCell *Battle::nearestCell(RuntimeShip &s, Vec2 p) {
    RuntimeCell *best = nullptr;
    double bestSq = std::numeric_limits<double>::infinity();
    for (auto &c : s.cells)
        if (c.alive) {
            Vec2 w = localToWorld(s.position, s.cosHeading, s.sinHeading, c.coord);
            double d = sq({w.x - p.x, w.z - p.z});
            if (d < bestSq) {
                bestSq = d;
                best = &c;
            }
        }
    return best;
}

void Battle::resolveHit(RuntimeShip &s, RuntimeCell &c, const Projectile &p) {
    effects_.push_back({Effect::Type::Impact, p.pos, s.index, c.id, p.kind, s.heading,
                        static_cast<int>(std::round(p.damage))});
    bool changed = false;
    if (p.crew > 0 && s.crew > 0) {
        s.crewLost += p.crew;
        effects_.push_back({Effect::Type::Crew, p.pos, s.index, c.id, p.kind});
        changed = true;
    }
    if (damageCell(s, c, p.damage, p.pierce))
        changed = true;
    if (changed)
        refresh(s);
}
bool Battle::damageCell(RuntimeShip &s, RuntimeCell &c, double amount, bool pierce) {
    if (!c.alive)
        return false;
    double soak = pierce ? std::floor(c.soak / 2.0) : c.soak;
    c.hp -= std::max(1.0, amount - soak) * hullDamage(hullIndex_) * damageScale_ * overtime(time_);
    if (c.hp > 0)
        return false;
    c.hp = 0;
    c.alive = false;
    --s.aliveCells;
    effects_.push_back({Effect::Type::Destroy, localToWorld(s.position, s.cosHeading, s.sinHeading, c.coord),
                        s.index, c.id, Ammo::Round, s.heading, static_cast<int>(std::round(amount))});
    log_.push_back({time_, s.index, std::string(part(c.id).name) + " destroyed"});
    if (c.id == PartId::Magazine) {
        effects_.push_back({Effect::Type::Detonate,
                            localToWorld(s.position, s.cosHeading, s.sinHeading, c.coord), s.index, c.id,
                            Ammo::Round, s.heading, 15});
        log_.push_back({time_, s.index, "magazine detonation"});
        for (int ox = -1; ox <= 1; ++ox)
            for (int oz = -1; oz <= 1; ++oz)
                if (ox || oz)
                    for (auto &n : s.cells)
                        if (n.coord == Coord{c.coord.dx + ox, c.coord.dz + oz} && n.alive) {
                            damageCell(s, n, 15, true);
                            break;
                        }
        s.crewLost += 2;
    }
    sever(s);
    return true;
}
void Battle::sever(RuntimeShip &s) {
    RuntimeCell *helm = nullptr;
    for (auto &c : s.cells)
        if (c.id == PartId::Helm && c.alive)
            helm = &c;
    if (!helm)
        return;
    std::vector<RuntimeCell *> stack{helm};
    std::uint32_t stamp = static_cast<std::uint32_t>(tickCount_ + 1);
    helm->reached = stamp;
    while (!stack.empty()) {
        auto *c = stack.back();
        stack.pop_back();
        for (int i = 0; i < 4; ++i) {
            Coord q{c->coord.dx + (i == 0   ? 1
                                   : i == 1 ? -1
                                            : 0),
                    c->coord.dz + (i == 2   ? 1
                                   : i == 3 ? -1
                                            : 0)};
            for (auto &n : s.cells)
                if (n.coord == q && n.alive && n.reached != stamp) {
                    n.reached = stamp;
                    stack.push_back(&n);
                    break;
                }
        }
    }
    for (auto &c : s.cells)
        if (c.alive && c.reached != stamp) {
            c.alive = false;
            c.hp = 0;
            --s.aliveCells;
            effects_.push_back({Effect::Type::Sever,
                                localToWorld(s.position, s.cosHeading, s.sinHeading, c.coord), s.index,
                                c.id});
        }
}

void Battle::stepProjectiles() {
    std::vector<Projectile> keep;
    keep.reserve(projectiles_.size());
    for (const auto &p : projectiles_) {
        Projectile q = p;
        q.ttl -= Tick;
        q.pos.x += q.velocity.x * Tick;
        q.pos.z += q.velocity.z * Tick;
        if (q.ttl <= 0) {
            effects_.push_back({Effect::Type::Splash, q.pos});
            continue;
        }
        if (q.target < 0 || q.target >= static_cast<int>(ships.size()))
            continue;
        auto &t = ships[q.target];
        Vec2 rel{q.pos.x - t.position.x, q.pos.z - t.position.z};
        double gap = sq(rel);
        if (gap <= t.hitRadiusSq) {
            if (auto *c = cellAt(t, rel)) {
                resolveHit(t, *c, q);
                continue;
            }
        } else if (q.velocity.x * rel.x + q.velocity.z * rel.z >= 0) {
            effects_.push_back({Effect::Type::Splash, q.pos});
            continue;
        }
        keep.push_back(q);
    }
    projectiles_.swap(keep);
}

void Battle::checkEnd() {
    bool struck = false, stalemate = true;
    for (auto &s : ships)
        if (!s.out) {
            RuntimeCell *helm = nullptr;
            for (auto &c : s.cells)
                if (c.id == PartId::Helm) {
                    helm = &c;
                    break;
                }
            if (!helm || !helm->alive || s.aliveCells == 0)
                struck = true;
            else if (s.canFire)
                stalemate = false;
        }
    if (struck) {
        for (auto &s : ships) {
            RuntimeCell *helm = nullptr;
            for (auto &c : s.cells)
                if (c.id == PartId::Helm) {
                    helm = &c;
                    break;
                }
            if (s.out || (helm && helm->alive && s.aliveCells > 0))
                continue;
            s.out = true;
            s.outAt = time_;
            --afloat_;
            log_.push_back({time_, s.index, "strikes her colours"});
            std::vector<Projectile> keep;
            for (const auto &p : projectiles_)
                if (p.target == s.index)
                    effects_.push_back({Effect::Type::Splash, p.pos});
                else
                    keep.push_back(p);
            projectiles_.swap(keep);
        }
    }
    if (afloat_ > 1 && time_ < rules_.battleCap && !(stalemate && time_ > 5))
        return;
    over_ = true;
    result_.placing.clear();
    std::vector<int> order;
    for (const auto &s : ships)
        order.push_back(s.index);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        const auto &left = ships[a];
        const auto &right = ships[b];
        if (left.out != right.out)
            return !left.out;
        if (left.out && left.outAt != right.outAt)
            return left.outAt > right.outAt;
        const double lf = structureFraction(left), rf = structureFraction(right);
        return lf != rf ? lf > rf : a < b;
    });
    result_.placing = order;
    if (afloat_ <= 1) {
        int survivor = -1;
        for (const auto &s : ships)
            if (!s.out)
                survivor = s.index;
        result_.winner = survivor;
        result_.reason = survivor < 0 ? "all ships strike" : "helm lost";
        return;
    }
    result_.winner = order.empty() ? -1 : order.front();
    if (order.size() > 1 &&
        structureFraction(ships[order[0]]) - structureFraction(ships[order[1]]) < rules_.drawMargin)
        result_.winner = -1;
    result_.reason = result_.winner < 0 ? "time-limit draw" : (stalemate ? "stalemate" : "time limit");
}
void Battle::tick() {
    if (over_)
        return;
    time_ += Tick;
    auto pick = [&](RuntimeShip &s) {
        int best = s.target >= 0 && !ships[s.target].out ? s.target : -1;
        double bestSq =
            best >= 0 ? sq({ships[best].position.x - s.position.x, ships[best].position.z - s.position.z}) *
                            rules_.targetSwitchMargin * rules_.targetSwitchMargin
                      : std::numeric_limits<double>::infinity();
        for (const auto &foe : ships)
            if (foe.index != s.index && !foe.out) {
                double d = sq({foe.position.x - s.position.x, foe.position.z - s.position.z});
                if (d < bestSq) {
                    best = foe.index;
                    bestSq = d;
                }
            }
        return best;
    };
    if (ships.size() > 2 && time_ >= retargetAt_) {
        retargetAt_ = time_ + rules_.retargetSeconds;
        for (auto &s : ships)
            if (!s.out)
                s.target = pick(s);
    }
    for (auto &s : ships)
        if (!s.out && (s.target < 0 || ships[s.target].out))
            s.target = pick(s);
    for (auto &s : ships)
        if (!s.out && s.target >= 0)
            steer(s, ships[s.target]);
    for (int i = 0; i < static_cast<int>(ships.size()); ++i)
        for (int j = i + 1; j < static_cast<int>(ships.size()); ++j) {
            auto &a = ships[i];
            auto &b = ships[j];
            if (a.out || b.out)
                continue;
            Vec2 d{b.position.x - a.position.x, b.position.z - a.position.z};
            double dist = length(d);
            if (dist == 0)
                dist = .001;
            double nx = d.x / dist, nz = d.z / dist,
                   limit = std::max(MinSeparation, reachAlong(a, nx, nz) + reachAlong(b, nx, nz));
            if (dist < limit) {
                double push = (limit - dist) * .5;
                a.position.x -= nx * push;
                a.position.z -= nz * push;
                b.position.x += nx * push;
                b.position.z += nz * push;
                int pair = i * static_cast<int>(ships.size()) + j;
                if (time_ >= contactAt_[pair]) {
                    contactAt_[pair] = time_ + .5;
                    double rvx = b.sinHeading * b.speed - a.sinHeading * a.speed,
                           rvz = -b.cosHeading * b.speed + a.cosHeading * a.speed,
                           rel = std::sqrt(rvx * rvx + rvz * rvz),
                           amount = 5 * std::min(1.0, rel / BaseSpeed);
                    if (amount >= .5) {
                        Vec2 contact{(a.position.x + b.position.x) * .5, (a.position.z + b.position.z) * .5};
                        RuntimeCell *ca = nearestCell(a, contact);
                        RuntimeCell *cb = nearestCell(b, contact);
                        effects_.push_back({Effect::Type::Impact, contact, a.index});
                        effects_.push_back({Effect::Type::Impact, contact, b.index});
                        if (ca && damageCell(a, *ca, amount, true))
                            refresh(a);
                        if (cb && damageCell(b, *cb, amount, true))
                            refresh(b);
                    }
                }
            }
        }
    for (auto &s : ships)
        if (!s.out)
            fire(s);
    stepProjectiles();
    checkEnd();
}
double Battle::structureFraction(const RuntimeShip &s) const {
    double left = 0;
    for (const auto &c : s.cells)
        if (c.alive)
            left += c.hp;
    return s.initialStructure ? left / s.initialStructure : 0;
}
void Battle::finish() {
    if (!over_) {
        time_ = rules_.battleCap;
        checkEnd();
    }
    for (auto &s : ships) {
        for (const auto &c : s.cells)
            if (c.alive)
                s.design.parts[c.coord].hp = c.hp;
        for (const auto &c : s.cells)
            if (!c.alive)
                erasePart(s.design, c.coord);
        if (!s.design.parts.contains({0, 0}))
            setPart(s.design, {0, 0}, {PartId::Helm, part(PartId::Helm).hp});
    }
}
} // namespace broadside::sim
