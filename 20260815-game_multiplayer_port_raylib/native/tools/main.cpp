#include "audio/audio.h"
#include "game/autobuild.h"
#include "game/bot.h"
#include "game/match.h"
#include "game/shipyard.h"
#include "net/authority.h"
#include "net/client.h"
#include "net/transport.h"
#include "presentation/controller.h"
#include "presentation/quality.h"
#include "reference/reference.h"
#include "sim/checksum.h"
#include "sim/timeline.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>

using namespace broadside;
namespace {
void require(bool condition, const std::string &message) {
    if (!condition)
        throw std::runtime_error(message);
}

std::vector<sim::Design> sampleDesigns(int count, int hullIndex, std::uint32_t seed) {
    std::vector<sim::Design> out;
    sim::Rng rng(seed);
    for (int seat = 0; seat < count; ++seat) {
        game::Shipyard yard(sim::makeDesign(), hullIndex, 56);
        game::autoBuild(yard, rng);
        out.push_back(yard.design());
    }
    return out;
}
sim::Battle runBattle(int count = 2, int hull = 0, std::uint32_t seed = 1) {
    auto designs = sampleDesigns(count, hull, seed ^ 0x55aau);
    sim::Battle battle(designs, hull, seed, .8f);
    sim::Timeline timeline(battle);
    while (!battle.over()) {
        for (int seat = 0; seat < count; ++seat)
            if ((battle.tickCount() + seat * 17) % 137 == 0)
                timeline.add({battle.tickCount() + 2, seat,
                              (battle.tickCount() / 137) % 2 ? sim::Ammo::Grape : sim::Ammo::Round});
        timeline.runTo(battle.tickCount() + 1);
    }
    battle.finish();
    return battle;
}
int cmdBench(int count) {
    auto designs = sampleDesigns(2, 2, 99);
    auto start = std::chrono::steady_clock::now();
    std::uint32_t sum = 0;
    for (int i = 0; i < count; ++i) {
        sim::Battle battle(designs, 2, static_cast<std::uint32_t>(i + 1), .4f);
        battle.advanceTicks(2400);
        battle.finish();
        sum ^= sim::checksum(battle);
    }
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "bench: " << count << " battles in " << std::fixed << std::setprecision(3) << seconds
              << "s (" << (count / seconds) << "/s), checksum=" << sum << '\n';
    return 0;
}

int cmdTest() {
    sim::Rng vectorRng(1);
    const std::uint32_t expected[] = {2693262067u, 11749833u, 2265367787u, 4213581821u, 4159151403u};
    for (auto value : expected)
        require(vectorRng.nextU32() == value, "RNG vector changed");
    require(sim::fsin(.7f) == static_cast<float>(std::sin(.7f)), "float32 sine collapse changed");
    require(sim::fatan2(.3f, -.2f) == static_cast<float>(std::atan2(.3f, -.2f)),
            "float32 atan2 collapse changed");
    auto designs = sampleDesigns(2, 0, 7);
    sim::Battle whole(designs, 0, 42, .2f), ticks(designs, 0, 42, .2f);
    whole.advance(12.34f);
    ticks.advanceTicks(whole.tickCount());
    require(sim::checksum(whole) == sim::checksum(ticks), "fixed tick determinism failed");
    sim::Battle melee(sampleDesigns(4, 2, 91), 2, 44, .5f);
    for (int seat = 0; seat < 4; ++seat)
        require(melee.state()[seat].target == (seat + 1) % 4, "initial melee targets are not round-robin");
    for (int count = 2; count <= 4; ++count) {
        auto battle = runBattle(count, 2, 100 + count);
        require(battle.over() && battle.result().placing.size() == static_cast<std::size_t>(count),
                "battle result invariant failed");
    }
    game::Shipyard yard(sim::makeDesign(), 0, 20);
    require(!yard.place({0, 0}, sim::PartId::Mast).ok, "occupied placement accepted");
    require(!yard.place({0, -1}, sim::PartId::GunDeck).ok, "spine broadside accepted");
    int purse = yard.scrap();
    require(yard.place({-1, 0}, sim::PartId::Timber).ok, "legal placement rejected");
    require(yard.remove({-1, 0}).ok && yard.scrap() == purse, "same-phase refund changed");
    sim::Design old = sim::makeDesign();
    old.parts[{-1, 0}] = {sim::PartId::Timber, sim::part(sim::PartId::Timber).hp};
    game::Shipyard persistent(old, 0, 20);
    require(persistent.remove({-1, 0}).ok && persistent.scrap() == 20, "old equipment received a refund");
    sim::Design damaged = sim::makeDesign();
    damaged.parts[{-1, 0}] = {sim::PartId::Timber, 1};
    damaged.parts[{1, 0}] = {sim::PartId::Timber, 2};
    game::Shipyard refit(damaged, 0, 1);
    require(refit.refit().ok, "partial refit rejected");
    require(refit.design().parts.at({-1, 0}).hp == sim::part(sim::PartId::Timber).hp,
            "worst damage was not refitted first");
    require(refit.design().parts.at({1, 0}).hp == 2, "partial refit overspent purse");
    sim::Battle replay(designs, 0, 8, .2f);
    sim::Timeline timeline(replay);
    require(timeline.add({4, 0, sim::Ammo::Grape}), "first input rejected");
    require(!timeline.add({4, 0, sim::Ammo::Grape}), "duplicate input accepted");
    timeline.runTo(20);
    require(replay.state()[0].ammo == sim::Ammo::Grape, "timeline input failed");
    game::Match final = game::makeMatch(1, 3);
    final.over = true;
    final.round = 5;
    final.scores = {2, 1, 1};
    require(game::winner(final) == 0, "unique final leader not selected");
    final.scores = {2, 2, 1};
    require(game::winner(final) == -1, "final tie not preserved");
    std::cout << "test: deterministic simulation, shipyard, melee, replay, and economy invariants passed\n";
    return 0;
}

std::string nativeGolden() {
    auto designs = sampleDesigns(2, 1, 11);
    std::ostringstream out;
    out << "golden-native v2\n";
    for (int i = 0; i < 9; ++i) {
        sim::Battle battle(designs, 1, 0x1000u + i, .3f);
        battle.advanceTicks(2400);
        battle.finish();
        out << i << ' ' << battle.winner() << ' ' << battle.tickCount() << ' ' << sim::checksum(battle);
        for (const auto &ship : battle.state())
            out << " ship" << ship.index << ':' << ship.out << ':' << ship.aliveCells << ':' << ship.crew;
        out << '\n';
    }
    return out.str();
}
int cmdGolden() {
    const std::string actual = nativeGolden();
    const std::filesystem::path path =
        std::filesystem::path(BROADSIDE_SOURCE_DIR) / "native/reference/native_golden.txt";
    std::ifstream input(path);
    if (input) {
        std::ostringstream expected;
        expected << input.rdbuf();
        if (expected.str() != actual) {
            std::cout << actual << std::flush;
            throw std::runtime_error("strict native golden fixture changed");
        }
    }
    std::cout << actual;
    return 0;
}
int cmdReference() {
    const auto path = std::filesystem::path(BROADSIDE_SOURCE_DIR) / "native/reference/js_golden.txt";
    auto expected = reference::load(path);
    std::vector<reference::SemanticRow> actual;
    actual.reserve(expected.size());
    for (const auto &row : expected) {
        auto left = game::archetypeDesign(row.left, row.hull, game::cumulativeBudget(row.hull));
        auto right = game::archetypeDesign(row.right, row.hull, game::cumulativeBudget(row.hull));
        sim::Battle battle({left, right}, row.hull, row.seed, (row.seed % 360) * (sim::Pi / 180.0));
        game::AmmoBot bot(battle, {0, 1}, sim::Rng(1));
        int guard = 0;
        while (!battle.over() && guard++ < 240) {
            bot.update(.25f);
            battle.advance(.25);
            battle.effects().clear();
        }
        battle.finish();
        actual.push_back({row.hull, row.left, row.right, row.seed, battle.winner(), battle.time(),
                          battle.structureFraction(battle.state()[0]),
                          battle.structureFraction(battle.state()[1])});
    }
    auto comparison = reference::compare(expected, actual);
    std::cout << "reference: " << actual.size() << " battles; winner=" << comparison.winnerMismatches
              << " time=" << comparison.timeMismatches << " structure=" << comparison.structureMismatches
              << " mismatches\n";
    if (!comparison.ok()) {
        int shown = 0;
        for (std::size_t i = 0; i < expected.size() && shown < 5; ++i)
            if (expected[i].winner != actual[i].winner ||
                std::abs(expected[i].seconds - actual[i].seconds) > 1.0 / 60.0 + .0005 ||
                std::abs(expected[i].leftStructure - actual[i].leftStructure) > .001 ||
                std::abs(expected[i].rightStructure - actual[i].rightStructure) > .001) {
                std::cout << "  " << expected[i].hull << ' ' << expected[i].left << '/' << expected[i].right
                          << ' ' << expected[i].seed << " expected " << expected[i].winner << ' '
                          << expected[i].seconds << ' ' << expected[i].leftStructure << '/'
                          << expected[i].rightStructure << " actual " << actual[i].winner << ' '
                          << actual[i].seconds << ' ' << actual[i].leftStructure << '/'
                          << actual[i].rightStructure << '\n';
                ++shown;
            }
    }
    require(comparison.ok(), "native semantic grid differs from JavaScript fixture");
    return 0;
}

struct NetworkRun {
    int messages = 0;
    std::uint32_t authoritySum = 0;
    std::vector<std::uint32_t> clientSums;
};
NetworkRun runNetwork(int players, bool virtualWire) {
    std::vector<std::shared_ptr<net::ITransport>> wires;
    for (int seat = 0; seat < players; ++seat) {
        if (virtualWire)
            wires.push_back(std::make_shared<net::VirtualTransport>(.035f, .018f, 100 + seat));
        else
            wires.push_back(std::make_shared<net::ImmediateTransport>());
    }
    int messages = 0;
    net::RoomAuthority room(
        77, players,
        [&](int seat, const net::Message &message) {
            ++messages;
            wires[seat]->send(message);
        },
        players - 1);
    for (int seat = 0; seat < players; ++seat) {
        auto sink = [&, seat](const net::Command &command) { room.command(seat, command); };
        if (auto wire = std::dynamic_pointer_cast<net::ImmediateTransport>(wires[seat]))
            wire->bind(sink);
        else
            std::dynamic_pointer_cast<net::VirtualTransport>(wires[seat])->bind(sink);
    }
    std::vector<std::unique_ptr<net::GameClient>> clients;
    for (auto &wire : wires)
        clients.push_back(std::make_unique<net::GameClient>(wire));
    room.start();
    for (auto &client : clients)
        client->update(0);
    float now = 0;
    int requestedSeat = -1, requestedPhase = -1;
    bool disconnected = false, reconnected = false, badClock = false;
    while (room.phase() != 4 && now < 260) {
        if (room.phase() == 1 && room.activeSeat() >= 0 &&
            (requestedPhase != 1 || requestedSeat != room.activeSeat())) {
            requestedPhase = 1;
            requestedSeat = room.activeSeat();
            clients[requestedSeat]->command({net::LockCommand{}});
        }
        if (room.phase() == 2) {
            requestedPhase = 2;
            if (!badClock && room.battle() && room.battle()->tickCount() > 90) {
                wires[0]->send(
                    {net::Audience::privateSeat(0),
                     net::ChecksumSync{room.battle()->tickCount(), sim::checksum(*room.battle()) ^ 1u}});
                wires[0]->send(
                    {net::Audience::privateSeat(0),
                     net::ChecksumSync{room.battle()->tickCount(), sim::checksum(*room.battle())}});
                badClock = true;
            }
            if (virtualWire && !disconnected && room.battle() && room.battle()->tickCount() > 150) {
                std::dynamic_pointer_cast<net::VirtualTransport>(wires[0])->setConnected(false);
                disconnected = true;
            }
            if (disconnected && !reconnected && room.battle() && room.battle()->tickCount() > 240) {
                auto wire = std::dynamic_pointer_cast<net::VirtualTransport>(wires[0]);
                wire->setConnected(true);
                room.connect(0);
                reconnected = true;
            }
        }
        if (room.phase() == 3 && requestedPhase != 3) {
            requestedPhase = 3;
            requestedSeat = -1;
            clients[0]->command({net::ContinueCommand{}});
        }
        now += 1.0f / 60.0f;
        room.update(now);
        for (auto &client : clients)
            client->update(1.0f / 60.0f);
        if (room.phase() != requestedPhase && room.phase() != 1)
            requestedPhase = room.phase();
    }
    require(room.phase() == 4, "network match did not finish");
    for (int seat = 0; seat < players; ++seat)
        room.connect(seat);
    for (int flush = 0; flush < 120; ++flush)
        for (auto &client : clients)
            client->update(1.0f / 60.0f);
    require(badClock, "network clock repair was not exercised");
    if (virtualWire)
        require(reconnected, "virtual reconnect was not exercised");
    NetworkRun run;
    run.messages = messages;
    run.authoritySum = sim::checksum(*room.battle());
    for (int seat = 0; seat < players; ++seat) {
        auto &client = clients[seat];
        require(client->state().scores == room.match().scores, "client score replica diverged");
        require(client->battle() != nullptr, "client lost battle replica");
        run.clientSums.push_back(sim::checksum(*client->battle()));
        if (run.clientSums.back() != run.authoritySum)
            throw std::runtime_error("client final checksum diverged (players=" + std::to_string(players) +
                                     ", virtual=" + std::to_string(virtualWire) +
                                     ", seat=" + std::to_string(seat) +
                                     ", authority=" + std::to_string(run.authoritySum) + "@" +
                                     std::to_string(room.battle()->tickCount()) +
                                     ", client=" + std::to_string(run.clientSums.back()) + "@" +
                                     std::to_string(client->battle()->tickCount()) +
                                     ", inputs=" + std::to_string(client->inputLog().size()) + ")");
    }
    return run;
}
int cmdNetcheck() {
    int messages = 0;
    for (int players : {2, 4})
        for (bool wire : {false, true})
            messages += runNetwork(players, wire).messages;
    std::cout << "netcheck: complete 2/4-player matches over immediate/virtual wires, repair and reconnect "
                 "passed; messages="
              << messages << '\n';
    return 0;
}

int cmdPlaythrough() {
    constexpr int players = 2;
    std::vector<std::shared_ptr<net::ImmediateTransport>> wires;
    for (int i = 0; i < players; ++i)
        wires.push_back(std::make_shared<net::ImmediateTransport>());
    net::RoomAuthority room(
        913, players, [&](int seat, const net::Message &message) { wires[seat]->send(message); }, 0);
    for (int seat = 0; seat < players; ++seat)
        wires[seat]->bind([&, seat](const net::Command &command) { room.command(seat, command); });
    std::vector<net::GameClient> clients;
    for (auto &wire : wires)
        clients.emplace_back(wire);
    room.start();
    for (auto &client : clients)
        client.update(0);
    float now = 0;
    int handledRound = -1, handledSeat = -1;
    bool removed = false, rerolled = false, refit = false;
    while (room.phase() != 4 && now < 260) {
        if (room.phase() == 1 && room.activeSeat() >= 0 &&
            (handledRound != room.round() || handledSeat != room.activeSeat())) {
            handledRound = room.round();
            handledSeat = room.activeSeat();
            auto &client = clients[handledSeat];
            const auto *yard = room.yard(handledSeat);
            require(yard, "active yard missing");
            sim::Coord empty{};
            bool found = false;
            for (auto cell : sim::hull(yard->hullIndex()).cells)
                if (!yard->design().parts.contains(cell)) {
                    empty = cell;
                    found = true;
                    break;
                }
            if (found) {
                client.command({net::PlaceCommand{empty, sim::PartId::Timber}});
                client.update(0);
                client.command({net::RemoveCommand{empty}});
                client.update(0);
                removed = true;
            }
            if (!rerolled && yard->scrap() >= 2) {
                client.command({net::RerollCommand{}});
                client.update(0);
                rerolled = true;
            }
            if (room.round() > 0) {
                client.command({net::RefitCommand{}});
                client.update(0);
                refit = true;
            }
            client.command({net::LockCommand{}});
            client.update(0);
        } else if (room.phase() == 2) {
            now += .25f;
            room.update(now);
            for (auto &client : clients)
                client.update(.25f);
            continue;
        } else if (room.phase() == 3) {
            for (auto &client : clients) {
                client.command({net::ContinueCommand{}});
                client.update(0);
            }
        }
        now += 1.0f / 60.0f;
        room.update(now);
        for (auto &client : clients)
            client.update(1.0f / 60.0f);
    }
    require(room.phase() == 4 && removed && rerolled && refit, "automated playthrough coverage incomplete");
    for (int seat = 0; seat < players; ++seat)
        require(clients[seat].state().seat == seat, "private welcome audience failed");

    net::RoomAuthority botsOnly(914, 4, [](int, const net::Message &) {}, 4);
    botsOnly.start();
    float botNow = 0;
    while (botsOnly.phase() != 4 && botNow < 360) {
        botNow += .25f;
        botsOnly.update(botNow);
    }
    require(botsOnly.phase() == 4 && botsOnly.match().over,
            "four-seat bots-only match did not reach match end autonomously");
    std::cout << "playthrough: two human seats covered shipyard actions and results; four bots completed a "
                 "full unattended match\n";
    return 0;
}
int cmdRendercheck() {
    for (auto [width, height] : {std::pair{1280, 720}, std::pair{960, 540}}) {
        auto layout = presentation::UiLayout::make(width, height);
        for (const auto &rect :
             {layout.lock, layout.reroll, layout.refit, layout.roundAmmo, layout.grapeAmmo})
            require(rect.x >= 0 && rect.y >= 0 && rect.x + rect.w <= width && rect.y + rect.h <= height &&
                        rect.w > 0 && rect.h > 0,
                    "responsive layout escaped the viewport");
    }
    presentation::AdaptiveQuality quality;
    for (int i = 0; i < 180; ++i)
        quality.sample(.03f);
    require(quality.level() > 0, "adaptive renderer did not reduce scale under sustained load");
    const float reduced = quality.scale();
    for (int i = 0; i < 900; ++i)
        quality.sample(.01f);
    require(quality.scale() > reduced, "adaptive renderer did not recover after sustained fast frames");
    std::cout << "rendercheck: responsive bounds and adaptive scales [1,.85,.72,.60,.50] passed\n";
    return 0;
}

int cmdAudio() {
    float worstPeak = 0, worstDc = 0, worstOnset = 0;
    auto inspect = [&](const std::vector<float> &samples, int channels) {
        float peak = 0, mean = 0;
        for (float sample : samples) {
            peak = std::max(peak, std::abs(sample));
            mean += sample;
        }
        mean /= static_cast<float>(samples.size());
        float onset = peak > 0 ? std::abs(samples.front()) / peak : 0;
        worstPeak = std::max(worstPeak, peak);
        worstDc = std::max(worstDc, std::abs(mean));
        worstOnset = std::max(worstOnset, onset);
        require(channels == 1 || samples.size() % 2 == 0, "invalid mixer channel layout");
    };
    for (int kind = 0; kind < audio::CueCount; ++kind)
        inspect(audio::synthesize(static_cast<audio::Cue>(kind), 48000, static_cast<std::uint32_t>(kind + 1)),
                1);
    inspect(audio::broadsideBurst(), 2);
    audio::MixerStats stats;
    inspect(audio::worstCaseBurst(48000, &stats), 2);
    require(worstPeak <= 1.0f, "audio clipping detected");
    require(worstDc <= .01f, "audio DC offset exceeded envelope");
    require(worstOnset <= .25f, "audio onset exceeded envelope");
    require(stats.queueDrops == 0 && stats.started == stats.submitted,
            "mixer lost events inside configured queue budget");
    std::cout << "audio-check: 14 isolated cues, sixteen-gun broadside, and 66-event spatial burst peak="
              << worstPeak << " dc=" << worstDc << " onset=" << worstOnset << " steals=" << stats.voiceSteals
              << '\n';
    return 0;
}
int cmdMatch() {
    auto match = game::makeMatch(1234, 2);
    while (!match.over) {
        game::beginRound(match);
        auto battle = runBattle(2, game::rounds()[match.round].hull, match.round + 99);
        game::recordResult(match, battle.result());
        std::cout << "round " << match.round << ": score=" << match.scores[0] << '-' << match.scores[1]
                  << '\n';
    }
    return 0;
}
int cmdParts() {
    for (auto name : {"brawler", "massed", "sniper", "harasser", "crusher", "mixed"}) {
        auto design = game::archetypeDesign(name, 0, 34);
        std::cout << name;
        for (auto c : design.order) {
            auto it = design.parts.find(c);
            if (it != design.parts.end())
                std::cout << ' ' << c.dx << ',' << c.dz << ':' << static_cast<int>(it->second.id);
        }
        std::cout << '\n';
    }
    return 0;
}
struct MeleeMeasure {
    int battles = 0, decisive = 0, draws = 0;
    double seconds = 0;
    std::array<int, 4> wins{};
};
MeleeMeasure measureMelee(int seats, int battles, sim::BattleRules rules = {}) {
    MeleeMeasure measure;
    measure.battles = battles;
    for (int gameIndex = 0; gameIndex < battles; ++gameIndex) {
        std::vector<sim::Design> designs;
        const auto design = game::archetypeDesign("crusher", 2, 108);
        for (int seat = 0; seat < seats; ++seat)
            designs.push_back(design);
        const std::uint32_t seed = static_cast<std::uint32_t>(gameIndex * 104729 + seats);
        sim::Battle battle({designs, 2, seed, (seed % 360) * sim::Pi / 180.0, rules});
        std::vector<int> botSeats;
        for (int seat = 0; seat < seats; ++seat)
            botSeats.push_back(seat);
        game::AmmoBot bot(battle, botSeats, sim::Rng(1));
        while (!battle.over()) {
            bot.update(.25f);
            battle.advance(.25);
        }
        battle.finish();
        measure.seconds += battle.time();
        measure.decisive += battle.time() < rules.battleCap - .1;
        measure.draws += battle.winner() < 0;
        if (battle.winner() >= 0)
            ++measure.wins[static_cast<std::size_t>(battle.winner())];
    }
    return measure;
}
int cmdMelee() {
    static constexpr double chi95[] = {0, 3.84, 5.99, 7.81};
    for (int seats : {3, 4}) {
        auto m = measureMelee(seats, 300);
        const double decisive = static_cast<double>(m.decisive) / m.battles,
                     expected = static_cast<double>(m.battles - m.draws) / seats;
        double chi = 0;
        for (int seat = 0; seat < seats; ++seat)
            chi += (m.wins[seat] - expected) * (m.wins[seat] - expected) / expected;
        std::cout << seats << " ships: " << m.battles << " battles mean=" << m.seconds / m.battles
                  << "s decisive=" << decisive * 100 << "% draws=" << m.draws << " chi2=" << chi
                  << " seat wins";
        for (int seat = 0; seat < seats; ++seat)
            std::cout << ' ' << m.wins[seat];
        std::cout << '\n' << std::flush;
        require(decisive >= .95, "melee decisive-rate envelope failed");
        require(chi <= chi95[seats - 1], "melee seat-bias envelope failed");
    }
    return 0;
}
int cmdAblate() {
    struct Variant {
        const char *name;
        sim::BattleRules rules;
    };
    sim::BattleRules noRetarget;
    noRetarget.retargetSeconds = 100;
    sim::BattleRules eager;
    eager.targetSwitchMargin = 1;
    for (const auto &variant :
         {Variant{"canonical", {}}, Variant{"no-retarget", noRetarget}, Variant{"no-switch-margin", eager}}) {
        auto m = measureMelee(4, 60, variant.rules);
        std::cout << std::left << std::setw(18) << variant.name << " mean=" << std::setw(6)
                  << std::setprecision(3) << m.seconds / m.battles
                  << " decisive=" << 100.0 * m.decisive / m.battles << "% draws=" << m.draws << '\n';
    }
    return 0;
}
} // namespace

int main(int argc, char **argv) {
    const std::string command = argc > 1 ? argv[1] : "test";
    try {
        if (command == "bench")
            return cmdBench(argc > 2 ? std::atoi(argv[2]) : 1000);
        if (command == "golden")
            return cmdGolden();
        if (command == "reference")
            return cmdReference();
        if (command == "parts")
            return cmdParts();
        if (command == "match" || command == "balance")
            return cmdMatch();
        if (command == "melee")
            return cmdMelee();
        if (command == "ablate")
            return cmdAblate();
        if (command == "netcheck")
            return cmdNetcheck();
        if (command == "playthrough")
            return cmdPlaythrough();
        if (command == "rendercheck")
            return cmdRendercheck();
        if (command == "audio-check")
            return cmdAudio();
        if (command == "watch")
            return cmdBench(100);
        if (command == "check" || command == "test")
            return cmdTest();
        std::cerr << "usage: broadside_tool "
                     "[test|bench|golden|reference|parts|match|watch|balance|melee|ablate|netcheck|"
                     "playthrough|rendercheck|audio-check]\n";
        return 2;
    } catch (const std::exception &error) {
        std::cerr << "broadside_tool: " << error.what() << '\n';
        return 1;
    }
}
