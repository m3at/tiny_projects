#pragma once

#include "sim/battle.h"
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace broadside::net {

struct ContinueCommand {};
struct PlaceCommand {
    sim::Coord coord{};
    sim::PartId part = sim::PartId::Timber;
};
struct RemoveCommand {
    sim::Coord coord{};
};
struct RefitCommand {};
struct RerollCommand {};
struct LockCommand {};
struct SetAmmoCommand {
    sim::Ammo ammo = sim::Ammo::Round;
};
struct RematchCommand {};
using CommandPayload = std::variant<ContinueCommand, PlaceCommand, RemoveCommand, RefitCommand, RerollCommand,
                                    LockCommand, SetAmmoCommand, RematchCommand>;
struct Command {
    CommandPayload payload;
};

struct Audience {
    int seat = -1;
    static Audience broadcast() {
        return {-1};
    }
    static Audience privateSeat(int seat) {
        return {seat};
    }
    bool isBroadcast() const {
        return seat < 0;
    }
};
struct Welcome {
    int seat = 0;
    int seats = 2;
};
struct RoomState {
    int phase = 0;
    int round = 0;
    int activeSeat = -1;
    std::vector<int> scores;
    std::vector<bool> locked;
};
struct RoundIntro {
    int round = 0;
    int hull = 0;
    float wind = 0;
    int buildSeconds = 0;
};
struct BuildState {
    int round = 0;
    int hull = 0;
    int purse = 0;
    float deadline = 0;
    sim::Design design;
    std::vector<sim::PartId> offer;
    std::vector<std::string> warnings;
};
struct PurseCorrection {
    int purse = 0;
    sim::Design design;
};
struct Offer {
    int purse = 0;
    std::vector<sim::PartId> parts;
};
struct LockedSeat {
    int seat = 0;
};
struct BattleStart {
    sim::BattleInit init;
    int startTick = 0;
};
struct StampedAmmo {
    int seat = 0;
    int tick = 0;
    sim::Ammo ammo = sim::Ammo::Round;
};
struct ChecksumSync {
    int tick = 0;
    std::uint32_t sum = 0;
};
struct ShipSummary {
    int seat = 0;
    int soundness = 0;
    int firingGuns = 0;
    int hands = 0;
    int powder = 0;
    int lostMasts = 0;
};
struct Result {
    int winner = -1;
    int matchWinner = -1;
    bool matchOver = false;
    std::vector<int> placing;
    std::vector<int> scores;
    std::string reason;
    std::vector<ShipSummary> ships;
    std::vector<sim::BattleLog> log;
};
struct Denial {
    std::string why;
    BuildState correction;
};
using MessagePayload = std::variant<Welcome, RoomState, RoundIntro, BuildState, PurseCorrection, Offer,
                                    LockedSeat, BattleStart, StampedAmmo, ChecksumSync, Result, Denial>;
struct Message {
    Audience audience = Audience::broadcast();
    MessagePayload payload;
};

std::string messageName(const Message &message);

} // namespace broadside::net
