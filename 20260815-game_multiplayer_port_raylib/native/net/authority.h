#pragma once

#include "game/match.h"
#include "game/shipyard.h"
#include "protocol.h"
#include "sim/battle.h"
#include "sim/timeline.h"
#include <functional>
#include <memory>

namespace broadside::net {
class RoomAuthority {
  public:
    using Emit = std::function<void(int, const Message &)>;
    RoomAuthority(std::uint32_t seed, int players, Emit emit, int bots = 0);
    void start();
    void connect(int connection);
    void update(float now);
    void command(int connection, const Command &command);
    int phase() const {
        return phase_;
    }
    int round() const {
        return match_.round;
    }
    int activeSeat() const {
        return activeSeat_;
    }
    const game::Match &match() const {
        return match_;
    }
    const sim::Battle *battle() const {
        return battle_.get();
    }
    const game::Shipyard *yard(int seat) const {
        return seat >= 0 && seat < static_cast<int>(yards_.size()) ? &yards_[seat] : nullptr;
    }

  private:
    void openBuild();
    void advanceBuild();
    void beginBattle();
    void finishBattle();
    void sendRoom();
    void sendBuild(int seat);
    void deny(int seat, std::string why);
    void emitAll(MessagePayload payload);
    std::vector<sim::PartId> makeOffer(int seat, int reroll) const;
    Emit emit_;
    game::Match match_;
    int bots_ = 0;
    std::vector<game::Shipyard> yards_;
    std::unique_ptr<sim::Battle> battle_;
    std::unique_ptr<sim::Timeline> timeline_;
    float now_ = 0, battleStarted_ = 0, buildUntil_ = 0, nextBotAt_ = 0, resultUntil_ = 0;
    int phase_ = 0, activeSeat_ = -1;
    std::vector<bool> locked_, continued_;
    std::vector<int> rerolls_;
    std::vector<sim::Ammo> desired_;
};
} // namespace broadside::net
