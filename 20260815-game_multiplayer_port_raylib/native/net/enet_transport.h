#pragma once
#include "transport.h"
#include <cstdint>
#include <memory>
namespace broadside::net {
class EnetTransport final : public Transport {
  public:
    explicit EnetTransport(std::uint16_t listenPort = 0);
    ~EnetTransport() override;
    bool valid() const;
    void send(const Message &) override;
    void sendCommand(const Command &) override;
    void update(float seconds) override;
    bool poll(Message &) override;
    float serverTime() const override;

  private:
    struct State;
    std::unique_ptr<State> state_;
};
} // namespace broadside::net
