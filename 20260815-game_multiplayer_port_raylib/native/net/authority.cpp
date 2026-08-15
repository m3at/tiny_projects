#include "authority.h"
#include "game/autobuild.h"
#include "game/bot.h"
#include "sim/checksum.h"
#include "sim/rng.h"
#include <algorithm>
#include <cmath>
#include <type_traits>

namespace broadside::net {
using namespace sim;

RoomAuthority::RoomAuthority(std::uint32_t seed, int players, Emit emit, int bots)
    : emit_(std::move(emit)), match_(game::makeMatch(seed, std::clamp(players, 2, MaxPlayers))),
      bots_(std::clamp(bots, 0, std::clamp(players, 2, MaxPlayers))), locked_(match_.players, false),
      continued_(match_.players, false), rerolls_(match_.players, 0), desired_(match_.players, Ammo::Round) {}

void RoomAuthority::emitAll(MessagePayload payload) {
    for (int seat = 0; seat < match_.players; ++seat)
        emit_(seat, {Audience::broadcast(), payload});
}
void RoomAuthority::sendRoom() {
    emitAll(RoomState{phase_, match_.round, activeSeat_, match_.scores, locked_});
}
std::vector<PartId> RoomAuthority::makeOffer(int seat, int reroll) const {
    std::vector<PartId> out = {PartId::Timber, PartId::Magazine, PartId::Crew};
    const auto &buyable = buyableParts();
    Rng rng(hashSeed({match_.seed, static_cast<std::uint32_t>(match_.round), static_cast<std::uint32_t>(seat),
                      static_cast<std::uint32_t>(reroll), 0x0ffe42u}));
    while (out.size() < 5) {
        PartId id = buyable[rng.integer(0, static_cast<int>(buyable.size() - 1))];
        if (std::find(out.begin(), out.end(), id) == out.end())
            out.push_back(id);
    }
    return out;
}
void RoomAuthority::sendBuild(int seat) {
    const auto &yard = yards_[seat];
    BuildState state{
        match_.round,  yard.hullIndex(), yard.scrap(),         seat == activeSeat_ ? buildUntil_ : 0.0f,
        yard.design(), yard.offer(),     yard.stats().warnings};
    emit_(seat, {Audience::privateSeat(seat), state});
}
void RoomAuthority::deny(int seat, std::string why) {
    const auto &yard = yards_[seat];
    BuildState state{
        match_.round,  yard.hullIndex(), yard.scrap(),         seat == activeSeat_ ? buildUntil_ : 0.0f,
        yard.design(), yard.offer(),     yard.stats().warnings};
    emit_(seat, {Audience::privateSeat(seat), Denial{std::move(why), std::move(state)}});
}
void RoomAuthority::start() {
    match_ = game::makeMatch(match_.seed, match_.players);
    phase_ = 0;
    for (int seat = 0; seat < match_.players; ++seat)
        connect(seat);
    game::beginRound(match_);
    openBuild();
}
void RoomAuthority::connect(int connection) {
    if (connection < 0 || connection >= match_.players)
        return;
    emit_(connection, {Audience::privateSeat(connection), Welcome{connection, match_.players}});
    emit_(connection, {Audience::privateSeat(connection),
                       RoomState{phase_, match_.round, activeSeat_, match_.scores, locked_}});
    if (phase_ == 1 && !yards_.empty())
        sendBuild(connection);
    if (phase_ >= 2 && battle_) {
        BattleInit init{match_.designs, battle_->hullIndex(), battle_->seed(), battle_->windTo(), {}};
        emit_(connection, {Audience::privateSeat(connection), BattleStart{init, 0}});
        if (timeline_)
            for (const auto &input : timeline_->inputs())
                emit_(connection,
                      {Audience::privateSeat(connection), StampedAmmo{input.seat, input.tick, input.ammo}});
        emit_(connection,
              {Audience::privateSeat(connection), ChecksumSync{battle_->tickCount(), checksum(*battle_)}});
    }
}
void RoomAuthority::openBuild() {
    phase_ = 1;
    battle_.reset();
    timeline_.reset();
    yards_.clear();
    locked_.assign(match_.players, false);
    continued_.assign(match_.players, false);
    rerolls_.assign(match_.players, 0);
    desired_.assign(match_.players, Ammo::Round);
    const auto round = game::rounds()[match_.round];
    const int humans = match_.players - bots_;
    for (int seat = 0; seat < match_.players; ++seat) {
        yards_.emplace_back(match_.designs[seat], round.hull, match_.scrap[seat]);
        if (seat >= humans) {
            Rng rng(hashSeed(
                {match_.seed, static_cast<std::uint32_t>(match_.round), static_cast<std::uint32_t>(seat)}));
            game::autoBuild(yards_.back(), rng);
            locked_[seat] = true;
        } else
            yards_.back().setOffer(makeOffer(seat, 0));
    }
    activeSeat_ = humans > 0 ? 0 : -1;
    buildUntil_ = now_ + round.buildSeconds;
    emitAll(RoundIntro{match_.round, round.hull, match_.wind, round.buildSeconds});
    for (int seat = 0; seat < humans; ++seat)
        sendBuild(seat);
    sendRoom();
    if (activeSeat_ < 0)
        beginBattle();
}
void RoomAuthority::advanceBuild() {
    const int humans = match_.players - bots_;
    activeSeat_ = -1;
    for (int seat = 0; seat < humans; ++seat)
        if (!locked_[seat]) {
            activeSeat_ = seat;
            break;
        }
    if (activeSeat_ < 0) {
        beginBattle();
        return;
    }
    buildUntil_ = now_ + game::rounds()[match_.round].buildSeconds;
    sendBuild(activeSeat_);
    sendRoom();
}
void RoomAuthority::beginBattle() {
    if (phase_ != 1)
        return;
    phase_ = 2;
    activeSeat_ = -1;
    std::vector<Design> designs;
    for (auto &yard : yards_)
        designs.push_back(yard.design());
    match_.designs = designs;
    const std::uint32_t seed = hashSeed({match_.seed, static_cast<std::uint32_t>(match_.round), 0xba771eu});
    BattleInit init{designs, game::rounds()[match_.round].hull, seed, match_.wind, {}};
    battle_ = std::make_unique<Battle>(init);
    timeline_ = std::make_unique<Timeline>(*battle_);
    battleStarted_ = now_;
    nextBotAt_ = now_;
    emitAll(BattleStart{init, 0});
    sendRoom();
}
void RoomAuthority::finishBattle() {
    if (!battle_ || phase_ != 2)
        return;
    battle_->finish();
    for (int seat = 0; seat < match_.players; ++seat)
        match_.designs[seat] = battle_->state()[seat].design;
    const auto battleResult = battle_->result();
    game::recordResult(match_, battleResult);
    phase_ = match_.over ? 4 : 3;
    resultUntil_ = now_ + 8.0f;
    continued_.assign(match_.players, false);
    std::vector<ShipSummary> summaries;
    for (const auto &ship : battle_->state()) {
        int firing = 0, initialMasts = 0;
        for (const auto &gun : ship.guns)
            firing += gun.manned && ship.cells[gun.cell].alive;
        for (const auto &cell : ship.cells)
            initialMasts += cell.id == PartId::Mast;
        summaries.push_back({ship.index, static_cast<int>(std::round(battle_->structureFraction(ship) * 100)),
                             firing, ship.crew, ship.magazines, initialMasts - ship.masts});
    }
    emitAll(Result{battleResult.winner, game::winner(match_), match_.over, battleResult.placing,
                   match_.scores, battleResult.reason, summaries, battle_->log()});
    sendRoom();
}
void RoomAuthority::update(float now) {
    now_ = now;
    if (phase_ == 1 && activeSeat_ >= 0 && now_ >= buildUntil_) {
        locked_[activeSeat_] = true;
        match_.designs[activeSeat_] = yards_[activeSeat_].design();
        match_.scrap[activeSeat_] = yards_[activeSeat_].scrap();
        emitAll(LockedSeat{activeSeat_});
        advanceBuild();
    }
    if (phase_ == 2 && battle_ && !battle_->over()) {
        if (now_ >= nextBotAt_) {
            nextBotAt_ = now_ + .25f;
            for (int seat = match_.players - bots_; seat < match_.players; ++seat) {
                const auto &ship = battle_->state()[seat];
                command(seat, {SetAmmoCommand{ship.target >= 0
                                                  ? game::chooseAmmo(ship, battle_->state()[ship.target])
                                                  : Ammo::Round}});
            }
        }
        timeline_->runTo(
            std::max(battle_->tickCount(), static_cast<int>((now_ - battleStarted_) / TickSeconds)));
        if (battle_->tickCount() > 0 && battle_->tickCount() % 30 == 0)
            emitAll(ChecksumSync{battle_->tickCount(), checksum(*battle_)});
    }
    if (phase_ == 2 && battle_ && battle_->over())
        finishBattle();
    if (phase_ == 3 && now_ >= resultUntil_) {
        game::beginRound(match_);
        openBuild();
    }
}
void RoomAuthority::command(int connection, const Command &command) {
    if (connection < 0 || connection >= match_.players)
        return;
    const int seat = connection;
    std::visit(
        [&](const auto &value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, SetAmmoCommand>) {
                if (phase_ != 2 || !battle_) {
                    return;
                }
                if (desired_[seat] == value.ammo)
                    return;
                desired_[seat] = value.ammo;
                int tick =
                    std::max(battle_->tickCount(), static_cast<int>((now_ - battleStarted_) / TickSeconds)) +
                    2;
                timeline_->add({tick, seat, value.ammo});
                emitAll(StampedAmmo{seat, tick, value.ammo});
            } else if constexpr (std::is_same_v<T, ContinueCommand>) {
                if (phase_ != 3)
                    return;
                continued_[seat] = true;
                const int humans = match_.players - bots_;
                if (std::all_of(continued_.begin(), continued_.begin() + humans,
                                [](bool ready) { return ready; })) {
                    game::beginRound(match_);
                    openBuild();
                }
            } else if constexpr (std::is_same_v<T, RematchCommand>) {
                if (phase_ == 4)
                    start();
            } else {
                if (phase_ != 1 || seat != activeSeat_ || locked_[seat]) {
                    if (phase_ == 1)
                        deny(seat, "It is not this captain's build turn.");
                    return;
                }
                auto &yard = yards_[seat];
                game::ActionResult result;
                if constexpr (std::is_same_v<T, PlaceCommand>)
                    result = yard.place(value.coord, value.part);
                else if constexpr (std::is_same_v<T, RemoveCommand>)
                    result = yard.remove(value.coord);
                else if constexpr (std::is_same_v<T, RefitCommand>)
                    result = yard.refit();
                else if constexpr (std::is_same_v<T, RerollCommand>) {
                    Rng rng(hashSeed({match_.seed, static_cast<std::uint32_t>(match_.round),
                                      static_cast<std::uint32_t>(seat),
                                      static_cast<std::uint32_t>(rerolls_[seat] + 1)}));
                    result = yard.reroll(rng);
                    if (result.ok) {
                        ++rerolls_[seat];
                        yard.setOffer(makeOffer(seat, rerolls_[seat]));
                        emit_(seat, {Audience::privateSeat(seat), Offer{yard.scrap(), yard.offer()}});
                    }
                } else if constexpr (std::is_same_v<T, LockCommand>) {
                    locked_[seat] = true;
                    match_.designs[seat] = yard.design();
                    match_.scrap[seat] = yard.scrap();
                    emitAll(LockedSeat{seat});
                    advanceBuild();
                    return;
                }
                if (!result.ok)
                    deny(seat, result.why);
                else
                    sendBuild(seat);
            }
        },
        command.payload);
}
} // namespace broadside::net
