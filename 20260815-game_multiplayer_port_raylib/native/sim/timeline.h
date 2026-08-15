#pragma once
#include "battle.h"
#include <functional>
#include <vector>

namespace broadside::sim {
struct AmmoInput {
    int tick = 0;
    int seat = 0;
    Ammo ammo = Ammo::Round;
};
class Timeline {
  public:
    explicit Timeline(Battle &battle) : battle_(battle) {}
    bool add(AmmoInput input);
    void runTo(int tick);
    void runToMarks(int tick, int every, const std::function<void(int, const Battle &)> &mark);
    void rewind() {
        cursor_ = 0;
    }
    const std::vector<AmmoInput> &inputs() const {
        return inputs_;
    }

  private:
    Battle &battle_;
    std::vector<AmmoInput> inputs_;
    std::size_t cursor_ = 0;
};
} // namespace broadside::sim
