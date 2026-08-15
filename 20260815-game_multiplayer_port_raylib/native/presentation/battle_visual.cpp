#include "battle_visual.h"

#include <algorithm>

namespace broadside::presentation {

bool BattleVisualState::update(const sim::Battle &battle, float frameSeconds) {
    const bool newBattle = !initialized_ || seed_ != battle.seed() || battle.tickCount() < lastTick_;
    if (newBattle) {
        sinkSeconds_.fill(0.0f);
        seed_ = battle.seed();
        initialized_ = true;
    }
    lastTick_ = battle.tickCount();
    frameSeconds = std::clamp(frameSeconds, 0.0f, .05f);
    for (const auto &ship : battle.state()) {
        auto &elapsed = sinkSeconds_[static_cast<std::size_t>(ship.index)];
        elapsed = ship.out ? std::min(SinkTime, elapsed + frameSeconds) : 0.0f;
    }
    return newBattle;
}

void BattleVisualState::reset() {
    sinkSeconds_.fill(0.0f);
    seed_ = 0;
    lastTick_ = -1;
    initialized_ = false;
}

float BattleVisualState::sinkProgress(int seat) const {
    if (seat < 0 || seat >= sim::MaxPlayers)
        return 0.0f;
    return std::clamp(sinkSeconds_[static_cast<std::size_t>(seat)] / SinkTime, 0.0f, 1.0f);
}

float BattleVisualState::sinkAmount(int seat) const {
    const float descent = std::clamp((sinkProgress(seat) - .08f) / .68f, 0.0f, 1.0f);
    return descent * descent * (3.0f - 2.0f * descent);
}

float BattleVisualState::sinkDepth(int seat) const {
    return SinkDrop * sinkAmount(seat);
}

bool BattleVisualState::visible(int seat) const {
    return sinkAmount(seat) < .995f;
}

} // namespace broadside::presentation
