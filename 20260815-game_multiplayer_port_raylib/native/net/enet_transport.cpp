#include "enet_transport.h"
#include "protocol.h"
#ifdef BROADSIDE_HAS_ENET
#include <enet/enet.h>
#endif

namespace broadside::net {
struct EnetTransport::State {
#ifdef BROADSIDE_HAS_ENET
    ENetHost *host = nullptr;
    ENetPeer *peer = nullptr;
#endif
    bool valid = false;
};

EnetTransport::EnetTransport(std::uint16_t listenPort) : state_(std::make_unique<State>()) {
#ifdef BROADSIDE_HAS_ENET
    if (enet_initialize() == 0) {
        ENetAddress address{};
        address.host = ENET_HOST_ANY;
        address.port = listenPort;
        state_->host = enet_host_create(&address, 4, 2, 0, 0);
        state_->valid = state_->host != nullptr;
    }
#else
    (void)listenPort;
#endif
}
EnetTransport::~EnetTransport() {
#ifdef BROADSIDE_HAS_ENET
    if (state_->host)
        enet_host_destroy(state_->host);
    enet_deinitialize();
#endif
}
bool EnetTransport::valid() const {
    return state_->valid;
}
void EnetTransport::send(const Message &message) {
#ifdef BROADSIDE_HAS_ENET
    if (!state_->peer)
        return;
    const auto name = messageName(message);
    ENetPacket *packet = enet_packet_create(name.data(), name.size(), ENET_PACKET_FLAG_RELIABLE);
    enet_peer_send(state_->peer, 0, packet);
#else
    (void)message;
#endif
}
void EnetTransport::update(float seconds) {
#ifdef BROADSIDE_HAS_ENET
    (void)seconds;
    if (!state_->host)
        return;
    ENetEvent event{};
    while (enet_host_service(state_->host, &event, 0) > 0) {
        if (event.type == ENET_EVENT_TYPE_CONNECT)
            state_->peer = event.peer;
        else if (event.type == ENET_EVENT_TYPE_RECEIVE) {
            enet_packet_destroy(event.packet);
        } else if (event.type == ENET_EVENT_TYPE_DISCONNECT && event.peer == state_->peer)
            state_->peer = nullptr;
    }
#else
    (void)seconds;
#endif
}
void EnetTransport::sendCommand(const Command &) {}
bool EnetTransport::poll(Message &) {
    return false;
}
float EnetTransport::serverTime() const {
    return 0.0f;
}
} // namespace broadside::net
