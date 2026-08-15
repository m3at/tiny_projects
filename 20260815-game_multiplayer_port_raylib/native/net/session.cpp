#include "session.h"
#include <algorithm>
namespace broadside::net {
LocalSession::LocalSession(std::uint32_t seed, int players, int bots)
    : bots_(std::clamp(bots, 0, std::clamp(players, 2, sim::MaxPlayers))) {
    players = std::clamp(players, 2, sim::MaxPlayers);
    for (int seat = 0; seat < players; ++seat)
        wires_.push_back(std::make_shared<ImmediateTransport>());
    authority_ = std::make_unique<RoomAuthority>(
        seed, players, [this](int seat, const Message &message) { wires_[seat]->send(message); }, bots_);
    for (int seat = 0; seat < players; ++seat) {
        wires_[seat]->bind([this, seat](const Command &command) { authority_->command(seat, command); });
        clients_.push_back(std::make_unique<GameClient>(wires_[seat]));
    }
    authority_->start();
    for (auto &client : clients_)
        client->update(0);
}
void LocalSession::update(float seconds) {
    clock_ += std::max(0.0f, seconds);
    authority_->update(clock_);
    for (auto &client : clients_)
        client->update(seconds);
}
} // namespace broadside::net
