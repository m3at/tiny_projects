#pragma once

#include "protocol.h"
#include <deque>
#include <functional>
#include <random>

namespace broadside::net {
using CommandSink = std::function<void(const Command &)>;

class ITransport {
  public:
    virtual ~ITransport() = default;
    virtual void send(const Message &message) = 0;
    virtual void sendCommand(const Command &command) = 0;
    virtual void update(float seconds) = 0;
    virtual bool poll(Message &message) = 0;
    virtual float serverTime() const = 0;
};
using Transport = ITransport;

class ImmediateTransport final : public ITransport {
  public:
    explicit ImmediateTransport(CommandSink commands = {}) : commands_(std::move(commands)) {}
    void bind(CommandSink commands) {
        commands_ = std::move(commands);
    }
    void send(const Message &message) override {
        messages_.push_back(message);
    }
    void sendCommand(const Command &command) override {
        if (commands_)
            commands_(command);
    }
    void update(float seconds) override {
        now_ += seconds;
    }
    bool poll(Message &message) override;
    float serverTime() const override {
        return now_;
    }

  private:
    CommandSink commands_;
    std::deque<Message> messages_;
    float now_ = 0;
};

class VirtualTransport final : public ITransport {
  public:
    VirtualTransport(float latency = 0.1f, float jitter = 0.02f, std::uint32_t seed = 1);
    void bind(CommandSink commands) {
        commands_ = std::move(commands);
    }
    void send(const Message &message) override;
    void sendCommand(const Command &command) override;
    void update(float seconds) override;
    bool poll(Message &message) override;
    float serverTime() const override {
        return now_;
    }
    void setConnected(bool connected) {
        connected_ = connected;
    }
    bool connected() const {
        return connected_;
    }
    void setReordering(bool enabled) {
        reorder_ = enabled;
    }

  private:
    template <class T> struct Pending {
        float at;
        std::uint64_t order;
        T value;
    };
    float delay();
    float now_ = 0, latency_, jitter_;
    std::uint64_t order_ = 0;
    bool connected_ = true, reorder_ = true;
    std::mt19937 rng_;
    std::uniform_real_distribution<float> dist_{-1, 1};
    CommandSink commands_;
    std::deque<Pending<Message>> incoming_;
    std::deque<Pending<Command>> outgoing_;
    std::deque<Message> ready_;
};

using LocalTransport = ImmediateTransport;
using MockTransport = VirtualTransport;
} // namespace broadside::net
