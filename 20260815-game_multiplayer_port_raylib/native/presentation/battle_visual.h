#pragma once

#include "sim/battle.h"

#include <array>
#include <cstdint>

namespace broadside::presentation {

// Presentation time continues while an authoritative battle is frozen for its verdict. Keeping
// this state outside both simulation and raylib makes round transitions and sinking deterministic
// to test without giving presentation any authority over the battle.
class BattleVisualState {
  public:
    static constexpr float SinkTime = 2.8f;
    static constexpr float SinkDrop = 2.4f;

    // Returns true when the renderer must atomically reset battle-local camera and effect state.
    bool update(const sim::Battle &battle, float frameSeconds);
    void reset();

    [[nodiscard]] float sinkProgress(int seat) const;
    [[nodiscard]] float sinkAmount(int seat) const;
    [[nodiscard]] float sinkDepth(int seat) const;
    [[nodiscard]] bool visible(int seat) const;

  private:
    std::array<float, sim::MaxPlayers> sinkSeconds_{};
    std::uint32_t seed_ = 0;
    int lastTick_ = -1;
    bool initialized_ = false;
};

} // namespace broadside::presentation
