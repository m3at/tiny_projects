#include "client.h"
#include "game/shipyard.h"
#include "sim/checksum.h"
#include <type_traits>

namespace broadside::net {
void GameClient::optimistic(const Command &command) {
    if (!state_.build || state_.phase != 1)
        return;
    auto &mirror = *state_.build;
    game::Shipyard yard(mirror.design, mirror.hull, mirror.purse);
    yard.setOffer(mirror.offer);
    std::visit(
        [&](const auto &value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, PlaceCommand>)
                yard.place(value.coord, value.part);
            else if constexpr (std::is_same_v<T, RemoveCommand>)
                yard.remove(value.coord);
            else if constexpr (std::is_same_v<T, RefitCommand>)
                yard.refit();
        },
        command.payload);
    mirror.design = yard.design();
    mirror.purse = yard.scrap();
    mirror.warnings = yard.stats().warnings;
}
void GameClient::command(const Command &command) {
    optimistic(command);
    transport_->sendCommand(command);
}
void GameClient::rebuild(int tick) {
    if (!battleInit_)
        return;
    battle_ = std::make_unique<sim::Battle>(*battleInit_);
    timeline_ = std::make_unique<sim::Timeline>(*battle_);
    for (const auto &input : inputs_)
        timeline_->add(input);
    timeline_->runTo(tick);
}
void GameClient::receive(const Message &message) {
    if (!message.audience.isBroadcast() && state_.seat >= 0 && message.audience.seat != state_.seat)
        return;
    std::visit(
        [&](const auto &value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, Welcome>) {
                state_.seat = value.seat;
                state_.seats = value.seats;
            } else if constexpr (std::is_same_v<T, RoomState>) {
                state_.phase = value.phase;
                state_.round = value.round;
                state_.activeSeat = value.activeSeat;
                state_.scores = value.scores;
                state_.locked = value.locked;
            } else if constexpr (std::is_same_v<T, RoundIntro>) {
                state_.round = value.round;
                state_.phase = 1;
                state_.result.reset();
                battle_.reset();
                timeline_.reset();
                inputs_.clear();
            } else if constexpr (std::is_same_v<T, BuildState>) {
                state_.build = value;
                state_.lastDenial.clear();
            } else if constexpr (std::is_same_v<T, PurseCorrection>) {
                if (state_.build) {
                    state_.build->purse = value.purse;
                    state_.build->design = value.design;
                }
            } else if constexpr (std::is_same_v<T, Offer>) {
                if (state_.build) {
                    state_.build->purse = value.purse;
                    state_.build->offer = value.parts;
                }
            } else if constexpr (std::is_same_v<T, LockedSeat>) {
                if (value.seat >= 0 && value.seat < static_cast<int>(state_.locked.size()))
                    state_.locked[value.seat] = true;
            } else if constexpr (std::is_same_v<T, BattleStart>) {
                if (battleInit_ && battle_ && battleInit_->seed == value.init.seed)
                    return;
                battleInit_ = value.init;
                inputs_.clear();
                tickCarry_ = 0;
                rebuild(value.startTick);
                state_.phase = 2;
                state_.build.reset();
            } else if constexpr (std::is_same_v<T, StampedAmmo>) {
                sim::AmmoInput input{value.tick, value.seat, value.ammo};
                bool duplicate = false;
                for (const auto &old : inputs_)
                    if (old.tick == input.tick && old.seat == input.seat)
                        duplicate = true;
                if (!duplicate) {
                    int current = battle_ ? battle_->tickCount() : 0;
                    inputs_.push_back(input);
                    if (timeline_)
                        timeline_->add(input);
                    if (value.tick < current)
                        rebuild(current);
                }
            } else if constexpr (std::is_same_v<T, ChecksumSync>) {
                checksum_ = value.sum;
                if (!battle_)
                    return;
                if (battle_->tickCount() < value.tick)
                    timeline_->runTo(value.tick);
                if (battle_->tickCount() == value.tick && sim::checksum(*battle_) != value.sum) {
                    rebuild(value.tick);
                    state_.checksumMismatch = sim::checksum(*battle_) != value.sum;
                } else
                    state_.checksumMismatch = false;
            } else if constexpr (std::is_same_v<T, Result>) {
                state_.result = value;
                state_.lastResult = value;
                state_.scores = value.scores;
                state_.phase = value.matchOver ? 4 : 3;
            } else if constexpr (std::is_same_v<T, Denial>) {
                state_.lastDenial = value.why;
                state_.build = value.correction;
            }
        },
        message.payload);
}
void GameClient::update(float dt) {
    transport_->update(dt);
    Message message;
    while (transport_->poll(message))
        receive(message);
    if (battle_ && timeline_ && !battle_->over()) {
        tickCarry_ += std::max(0.0f, dt);
        int ticks = static_cast<int>(tickCarry_ / sim::TickSeconds);
        tickCarry_ -= ticks * sim::TickSeconds;
        timeline_->runTo(battle_->tickCount() + ticks);
    }
}
} // namespace broadside::net
