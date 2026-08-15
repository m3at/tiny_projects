#include "timeline.h"
#include <algorithm>
#include <functional>

namespace broadside::sim {
bool Timeline::add(AmmoInput input) {
    for (const auto &i : inputs_)
        if (i.tick == input.tick && i.seat == input.seat)
            return false;
    inputs_.push_back(input);
    std::sort(inputs_.begin(), inputs_.end(),
              [](auto &a, auto &b) { return a.tick != b.tick ? a.tick < b.tick : a.seat < b.seat; });
    if (cursor_ > inputs_.size())
        cursor_ = inputs_.size();
    return true;
}
void Timeline::runTo(int tick) {
    while (battle_.tickCount() < tick && !battle_.over()) {
        while (cursor_ < inputs_.size() && inputs_[cursor_].tick <= battle_.tickCount()) {
            const auto &i = inputs_[cursor_++];
            battle_.setAmmo(i.seat, i.ammo);
        }
        battle_.advanceTicks(1);
    }
    while (cursor_ < inputs_.size() && inputs_[cursor_].tick < battle_.tickCount())
        ++cursor_;
}
void Timeline::runToMarks(int tick, int every, const std::function<void(int, const Battle &)> &mark) {
    while (battle_.tickCount() < tick && !battle_.over()) {
        int next = ((battle_.tickCount() / every) + 1) * every;
        int before = battle_.tickCount();
        runTo(std::min(tick, next));
        if (before == battle_.tickCount())
            break;
        if (battle_.tickCount() % every == 0)
            mark(battle_.tickCount(), battle_);
    }
}
} // namespace broadside::sim
