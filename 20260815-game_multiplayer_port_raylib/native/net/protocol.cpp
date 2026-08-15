#include "protocol.h"

namespace broadside::net {
std::string messageName(const Message &message) {
    return std::visit(
        [](const auto &value) -> std::string {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, Welcome>)
                return "welcome";
            else if constexpr (std::is_same_v<T, RoomState>)
                return "room-state";
            else if constexpr (std::is_same_v<T, RoundIntro>)
                return "round-intro";
            else if constexpr (std::is_same_v<T, BuildState>)
                return "build-state";
            else if constexpr (std::is_same_v<T, PurseCorrection>)
                return "purse-correction";
            else if constexpr (std::is_same_v<T, Offer>)
                return "offer";
            else if constexpr (std::is_same_v<T, LockedSeat>)
                return "locked-seat";
            else if constexpr (std::is_same_v<T, BattleStart>)
                return "battle-start";
            else if constexpr (std::is_same_v<T, StampedAmmo>)
                return "stamped-ammo";
            else if constexpr (std::is_same_v<T, ChecksumSync>)
                return "checksum-sync";
            else if constexpr (std::is_same_v<T, Result>)
                return "result";
            else
                return "denial";
        },
        message.payload);
}
} // namespace broadside::net
