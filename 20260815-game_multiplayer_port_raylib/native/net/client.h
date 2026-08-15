#pragma once

#include "protocol.h"
#include "sim/battle.h"
#include "sim/timeline.h"
#include "transport.h"
#include <memory>
#include <optional>

namespace broadside::net {
struct ClientState {
    int phase = 0, seat = -1, seats = 0, round = 0, activeSeat = -1;
    std::vector<int> scores;
    std::vector<bool> locked;
    std::optional<BuildState> build;
    std::optional<Result> result;
    std::optional<Result> lastResult;
    std::string lastDenial;
    bool checksumMismatch = false;
};

class GameClient {
  public:
    explicit GameClient(std::shared_ptr<ITransport> transport) : transport_(std::move(transport)) {}
    void update(float dt);
    void command(const Command &command);
    const ClientState &state() const {
        return state_;
    }
    const sim::Battle *battle() const {
        return battle_.get();
    }
    std::uint32_t lastChecksum() const {
        return checksum_;
    }
    const std::vector<sim::AmmoInput> &inputLog() const {
        return inputs_;
    }

  private:
    void receive(const Message &message);
    void rebuild(int tick);
    void optimistic(const Command &command);
    std::shared_ptr<ITransport> transport_;
    ClientState state_;
    std::optional<sim::BattleInit> battleInit_;
    std::unique_ptr<sim::Battle> battle_;
    std::unique_ptr<sim::Timeline> timeline_;
    std::vector<sim::AmmoInput> inputs_;
    std::uint32_t checksum_ = 0;
    float tickCarry_ = 0;
};
using ClientReplica = GameClient;
} // namespace broadside::net
