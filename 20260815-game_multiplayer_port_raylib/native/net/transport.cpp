#include "transport.h"
#include <algorithm>

namespace broadside::net {
bool ImmediateTransport::poll(Message &message) {
    if (messages_.empty())
        return false;
    message = std::move(messages_.front());
    messages_.pop_front();
    return true;
}

VirtualTransport::VirtualTransport(float latency, float jitter, std::uint32_t seed)
    : latency_(std::max(0.0f, latency)), jitter_(std::max(0.0f, jitter)), rng_(seed) {}
float VirtualTransport::delay() {
    return std::max(0.0f, latency_ + jitter_ * dist_(rng_));
}
void VirtualTransport::send(const Message &message) {
    if (connected_)
        incoming_.push_back({now_ + delay(), order_++, message});
}
void VirtualTransport::sendCommand(const Command &command) {
    if (connected_)
        outgoing_.push_back({now_ + delay(), order_++, command});
}
void VirtualTransport::update(float seconds) {
    now_ += std::max(0.0f, seconds);
    if (!connected_)
        return;
    auto due = [](const auto &item, float now) { return item.at <= now; };
    std::vector<Pending<Command>> commands;
    for (auto it = outgoing_.begin(); it != outgoing_.end();)
        if (due(*it, now_)) {
            commands.push_back(std::move(*it));
            it = outgoing_.erase(it);
        } else
            ++it;
    std::vector<Pending<Message>> messages;
    for (auto it = incoming_.begin(); it != incoming_.end();)
        if (due(*it, now_)) {
            messages.push_back(std::move(*it));
            it = incoming_.erase(it);
        } else
            ++it;
    auto order = [&](const auto &a, const auto &b) {
        return reorder_ ? (a.at != b.at ? a.at < b.at : a.order > b.order) : a.order < b.order;
    };
    std::sort(commands.begin(), commands.end(), order);
    std::sort(messages.begin(), messages.end(), order);
    for (const auto &item : commands)
        if (commands_)
            commands_(item.value);
    for (auto &item : messages)
        ready_.push_back(std::move(item.value));
}
bool VirtualTransport::poll(Message &message) {
    if (ready_.empty())
        return false;
    message = std::move(ready_.front());
    ready_.pop_front();
    return true;
}
} // namespace broadside::net
