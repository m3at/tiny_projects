#pragma once
#include "authority.h"
#include "client.h"
#include "transport.h"
#include <memory>
#include <vector>
namespace broadside::net {
class LocalSession {
  public:
    LocalSession(std::uint32_t seed, int players, int bots);
    void update(float seconds);
    int players() const {
        return static_cast<int>(clients_.size());
    }
    int bots() const {
        return bots_;
    }
    int humans() const {
        return players() - bots_;
    }
    float serverTime() const {
        return clock_;
    }
    GameClient &client(int seat) {
        return *clients_.at(seat);
    }
    const GameClient &client(int seat) const {
        return *clients_.at(seat);
    }
    const ClientState &state() const {
        return clients_.front()->state();
    }
    const sim::Battle *battle() const {
        return clients_.front()->battle();
    }

  private:
    int bots_ = 0;
    float clock_ = 0;
    std::vector<std::shared_ptr<ImmediateTransport>> wires_;
    std::unique_ptr<RoomAuthority> authority_;
    std::vector<std::unique_ptr<GameClient>> clients_;
};
} // namespace broadside::net
