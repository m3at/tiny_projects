#include "bot.h"
#include <algorithm>
namespace broadside::game {
sim::Ammo chooseAmmo(const sim::RuntimeShip &me, const sim::RuntimeShip &enemy) {
    int manned = 0;
    for (const auto &gun : enemy.guns)
        if (enemy.cells[gun.cell].alive && gun.manned)
            ++manned;
    if (enemy.crew <= 0 || manned == 0)
        return sim::Ammo::Round;
    double best = 0;
    for (const auto &gun : me.guns)
        best = std::max(best, gun.spec.roundDamage);
    return enemy.crew <= 6 || best <= 2 ? sim::Ammo::Grape : sim::Ammo::Round;
}
void AmmoBot::update(float dt) {
    next_ -= dt;
    if (next_ > 0)
        return;
    next_ = .25f;
    for (int seat : seats_) {
        if (seat < 0 || seat >= battle_.shipCount())
            continue;
        const auto &s = battle_.state()[seat];
        int target = s.target;
        if (target >= 0)
            battle_.setAmmo(seat, chooseAmmo(s, battle_.state()[target]));
    }
}
} // namespace broadside::game
