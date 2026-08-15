#ifdef BROADSIDE_HAS_RAYLIB
#include "audio/audio.h"
#include "game/shipyard.h"
#include "presentation/controller.h"
#include "raylib.h"
#include "render/renderer.h"
#include "render/ui.h"
#include "rlgl.h"
#include "sim/checksum.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace broadside;
namespace {
constexpr Color Sea{16, 41, 52, 255};
constexpr Color Ink = render::theme::Ink;
constexpr Color Gold = render::theme::Gold;
constexpr Color Red = render::theme::Sinking;
Color shipColor(int seat) {
    return render::theme::Players[seat % 4];
}

const char *windName(double windTo) {
    static constexpr const char *names[]{"Northerly", "North-easterly", "Easterly", "South-easterly",
                                         "Southerly", "South-westerly", "Westerly", "North-westerly"};
    double from = std::fmod(windTo + PI + 2 * PI, 2 * PI);
    return names[static_cast<int>(std::round(from / (2 * PI) * 8)) % 8];
}
const char *partBlurb(sim::PartId id) {
    static constexpr const char *blurbs[]{
        "Filler. Shields spine.", "Armour. Soaks hits.",   "Adds 3 hands.",       "Speed and turning.",
        "Gunpowder. Explosive.",  "Light, all-round gun.", "3 medium side guns.", "2 close side guns.",
        "Piercing bow gun.",      "Fixed heart of ship."};
    return blurbs[static_cast<int>(id)];
}
void autoFitClient(net::GameClient &client) {
    if (!client.state().build)
        return;
    const auto offered = client.state().build->offer;
    const auto has = [&](sim::PartId id) {
        return std::find(offered.begin(), offered.end(), id) != offered.end();
    };
    const auto placeOne = [&](sim::PartId id) {
        const auto *current = client.state().build ? &*client.state().build : nullptr;
        if (!current || !has(id) || current->purse < sim::part(id).cost)
            return;
        for (const auto cell : sim::hull(current->hull).cells) {
            if (current->design.parts.contains(cell))
                continue;
            const auto &part = sim::part(id);
            if (part.gunSpec.arc == sim::Arc::Side && cell.dx == 0)
                continue;
            if (part.gunSpec.arc == sim::Arc::Bow && !sim::isBowCell(current->hull, cell.dz))
                continue;
            client.command({net::PlaceCommand{cell, id}});
            return;
        }
    };
    placeOne(sim::PartId::Magazine);
    placeOne(sim::PartId::Crew);
    placeOne(sim::PartId::Mast);
    const auto gun =
        std::find_if(offered.begin(), offered.end(), [](sim::PartId id) { return sim::part(id).gun; });
    if (gun != offered.end()) {
        placeOne(*gun);
        placeOne(*gun);
    }
    placeOne(sim::PartId::Crew);
    placeOne(sim::PartId::Mast);
    while (client.state().build && client.state().build->purse >= 1) {
        const auto used = client.state().build->design.parts.size();
        placeOne(sim::PartId::Timber);
        if (!client.state().build || client.state().build->design.parts.size() == used)
            break;
    }
}
class AudioBank {
  public:
    AudioBank() {
        InitAudioDevice();
        ready_ = IsAudioDeviceReady();
        if (!ready_)
            return;
        for (int index = 0; index < static_cast<int>(sounds_.size()); ++index) {
            auto samples = audio::synthesize(static_cast<audio::Cue>(index), 48000,
                                             static_cast<std::uint32_t>(index + 1));
            Wave wave{};
            wave.frameCount = static_cast<unsigned>(samples.size());
            wave.sampleRate = 48000;
            wave.sampleSize = 32;
            wave.channels = 1;
            wave.data = samples.data();
            sounds_[index] = LoadSoundFromWave(wave);
            SetSoundVolume(sounds_[index], .65f);
            for (auto &voice : voices_[index])
                voice = LoadSoundAlias(sounds_[index]);
        }
    }
    ~AudioBank() {
        if (!ready_)
            return;
        for (auto &pool : voices_)
            for (auto voice : pool)
                UnloadSoundAlias(voice);
        for (auto sound : sounds_)
            UnloadSound(sound);
        CloseAudioDevice();
    }
    void play(audio::Cue cue, bool muted, float pan = 0) {
        if (!ready_ || muted)
            return;
        const int index = static_cast<int>(cue);
        auto &pool = voices_[index];
        auto &voice = pool[next_[index]++ % pool.size()];
        SetSoundPan(voice, std::clamp(.5f + pan * .5f, 0.0f, 1.0f));
        SetSoundVolume(voice, cue == audio::Cue::Detonation ? .48f : .65f);
        PlaySound(voice);
    }
    void ambience(bool muted) {
        if (!ready_ || muted) {
            if (ready_)
                StopSound(sounds_[static_cast<int>(audio::Cue::Sea)]);
            return;
        }
        auto &sea = sounds_[static_cast<int>(audio::Cue::Sea)];
        if (!IsSoundPlaying(sea))
            PlaySound(sea);
    }

  private:
    bool ready_ = false;
    std::array<Sound, audio::CueCount> sounds_{};
    std::array<std::array<Sound, 4>, audio::CueCount> voices_{};
    std::array<std::size_t, audio::CueCount> next_{};
};
} // namespace

int main(int argc, char **argv) {
    int players = 2, bots = 1, windowWidth = 1280, windowHeight = 720;
    float speed = 1;
    std::uint32_t seed = 0xb0ad51deu;
    bool configured = false, botsOnly = false, loopMatches = false;
    std::string capturePath, scenario, baselinePath;
    int captureWarmupFrames = 0;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto value = [&]() { return i + 1 < argc ? argv[++i] : ""; };
        if (arg == "--players") {
            players = std::atoi(value());
            configured = true;
        } else if (arg == "--bots") {
            bots = std::atoi(value());
            configured = true;
        } else if (arg == "--bots-only") {
            botsOnly = true;
            configured = true;
        } else if (arg == "--loop") {
            loopMatches = true;
        } else if (arg == "--seed")
            seed = static_cast<std::uint32_t>(std::strtoul(value(), nullptr, 0));
        else if (arg == "--speed")
            speed = std::clamp(static_cast<float>(std::atof(value())), .25f, 8.0f);
        else if (arg == "--capture")
            capturePath = value();
        else if (arg == "--scenario")
            scenario = value();
        else if (arg == "--baseline-file")
            baselinePath = value();
        else if (arg == "--width")
            windowWidth = std::max(640, std::atoi(value()));
        else if (arg == "--height")
            windowHeight = std::max(360, std::atoi(value()));
        else if (arg == "--round")
            value();
    }
    players = std::clamp(players, 2, 4);
    bots = botsOnly ? players : std::clamp(bots, 0, players);
    if (!capturePath.empty())
        SetTraceLogLevel(LOG_WARNING);
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT |
                   (capturePath.empty() ? 0 : FLAG_WINDOW_HIDDEN));
    InitWindow(windowWidth, windowHeight, "Broadside");
    SetTargetFPS(60);
    SetExitKey(KEY_NULL);
    presentation::AppController app;
    if (configured)
        app.start(seed, players, bots);
    sim::PartId selected = sim::PartId::Timber;
    bool removeMode = false;
    bool debugOverlay = false;
    bool handoff = false;
    int lastBuildSeat = -1;
    int menuSelection = 0;
    float matchEndSeenAt = -1;
    bool muted = false;
    AudioBank audioBank;
    render::DesktopRenderer renderer;
    std::uint32_t effectSeed = 0;
    std::size_t effectCursor = 0;
    if (!scenario.empty() && scenario != "menu") {
        if (scenario == "four-way")
            app.start(seed, 4, 3);
        else if (scenario == "duel" || scenario == "sinking" || scenario == "result" ||
                 scenario == "match-end")
            app.start(seed, 2, 1);
        else
            app.start(seed, 4, 1);
        auto lockActive = [&]() {
            auto model = app.model();
            if (model.phase == 1 && model.activeSeat >= 0 && model.activeSeat < app.session().humans()) {
                auto &client = app.session().client(model.activeSeat);
                autoFitClient(client);
                client.command({net::LockCommand{}});
                app.update(.001f);
            }
        };
        if (scenario == "duel" || scenario == "sinking" || scenario == "four-way" || scenario == "result" ||
            scenario == "match-end") {
            int guard = 0;
            while (app.model().phase == 1 && guard++ < 12)
                lockActive();
            if ((scenario == "duel" || scenario == "four-way") && app.model().phase == 2)
                app.update(6.0f);
            if (scenario == "sinking") {
                guard = 0;
                auto shipIsOut = [&]() {
                    const auto *battle = app.session().battle();
                    return battle && std::any_of(battle->state().begin(), battle->state().end(),
                                                 [](const auto &ship) { return ship.out; });
                };
                while (app.model().phase == 2 && !shipIsOut() && guard++ < 800)
                    app.update(.05f);
                captureWarmupFrames = 96;
            }
            if (scenario == "result") {
                while (app.model().phase == 2 && guard++ < 400)
                    app.update(.25f);
            }
            if (scenario == "match-end") {
                guard = 0;
                while (app.model().phase != 4 && guard++ < 2400) {
                    auto model = app.model();
                    if (model.phase == 1)
                        lockActive();
                    else if (model.phase == 2)
                        app.update(.25f);
                    else if (model.phase == 3) {
                        for (int seat = 0; seat < app.session().humans(); ++seat)
                            app.session().client(seat).command({net::ContinueCommand{}});
                        app.update(.001f);
                    }
                }
            }
        }
    }
    int exitCode = 0;
    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_F11))
            ToggleFullscreen();
        if (IsKeyPressed(KEY_M))
            muted = !muted;
        if (IsKeyPressed(KEY_F3))
            debugOverlay = !debugOverlay;
        if (IsKeyPressed(KEY_ESCAPE) && app.started())
            app = presentation::AppController{};
        float dt = GetFrameTime() * speed;
        const bool presentationOnlyWarmup =
            !capturePath.empty() && scenario == "sinking" && captureWarmupFrames > 0;
        if (app.started() && !presentationOnlyWarmup)
            app.update(dt);
        if (app.started() && loopMatches && app.session().humans() == 0) {
            if (app.model().phase == 4) {
                if (matchEndSeenAt < 0)
                    matchEndSeenAt = app.session().serverTime();
                if (app.session().serverTime() - matchEndSeenAt >= 8.0f) {
                    app.session().client(0).command({net::RematchCommand{}});
                    app.update(.001f);
                    matchEndSeenAt = -1;
                }
            } else {
                matchEndSeenAt = -1;
            }
        }
        audioBank.ambience(muted);
        if (app.started() && app.session().battle()) {
            const auto *battle = app.session().battle();
            if (effectSeed != battle->seed()) {
                effectSeed = battle->seed();
                effectCursor = 0;
            }
            const auto &effects = battle->effects();
            for (int played = 0; effectCursor < effects.size() && played < 8; ++effectCursor, ++played) {
                using Type = sim::Effect::Type;
                auto cue = audio::Cue::Splinter;
                switch (effects[effectCursor].type) {
                case Type::Muzzle:
                    cue = audio::Cue::Cannon;
                    break;
                case Type::Impact:
                    cue = audio::Cue::RoundImpact;
                    break;
                case Type::Splash:
                    cue = audio::Cue::Splash;
                    break;
                case Type::Destroy:
                case Type::Sever:
                    cue = audio::Cue::Splinter;
                    break;
                case Type::Crew:
                    cue = audio::Cue::GrapeImpact;
                    break;
                case Type::Detonate:
                    cue = audio::Cue::Detonation;
                    break;
                case Type::Ammo:
                    cue = audio::Cue::Select;
                    break;
                }
                audioBank.play(
                    cue, muted,
                    std::clamp(static_cast<float>(effects[effectCursor].pos.x / 60.0), -1.0f, 1.0f));
            }
        }
        const int width = GetScreenWidth(), height = GetScreenHeight();
        render::UiPainter painter(renderer.headingFont(), renderer.bodyFont(), width, height);
        const float uiScale = painter.scale();
        const float logicalWidth = width / uiScale;
        const float logicalHeight = height / uiScale;
        BeginDrawing();
        ClearBackground(Sea);
        const auto currentModel = app.model();
        if (app.started() && currentModel.phase == 1 && currentModel.activeSeat >= 0 &&
            currentModel.activeSeat != lastBuildSeat) {
            lastBuildSeat = currentModel.activeSeat;
            handoff = capturePath.empty();
            removeMode = false;
        } else if (!app.started() || currentModel.phase != 1) {
            lastBuildSeat = -1;
            handoff = false;
        }
        if (app.started() && currentModel.phase == 2 && app.session().battle())
            renderer.drawBattle(*app.session().battle(), width, height,
                                capturePath.empty() ? GetFrameTime() : 1.0f / 60.0f);
        else
            renderer.drawSea(width, height, capturePath.empty() ? static_cast<float>(GetTime()) : 3.5f, .7);
        if (!app.started()) {
            const float centre = logicalWidth * .5f;
            if (IsKeyPressed(KEY_UP) || IsKeyPressed(KEY_DOWN))
                menuSelection = 1 - menuSelection;
            const int direction = IsKeyPressed(KEY_LEFT) ? -1 : IsKeyPressed(KEY_RIGHT) ? 1 : 0;
            if (direction && menuSelection == 0)
                players = std::clamp(players + direction, 2, 4);
            if (direction && menuSelection == 1)
                bots = std::clamp(bots + direction, 0, players - 1);
            painter.text("BROADSIDE", centre - painter.measure("BROADSIDE", 72, true) / uiScale * .5f, 57, 72,
                         Gold, true);
            painter.text("A LOCAL NAVAL COMMAND", centre - 102, 132, 12, render::theme::Dim, true);
            painter.panel(painter.rect(centre - 224, 188, 448, 358));
            painter.text("MUSTER YOUR FLEET", centre - 184, 220, 24, Ink, true);
            painter.text("Two to four ships. At least one captain must be human.", centre - 184, 254, 13,
                         render::theme::Dim);
            painter.section("FLEET", centre - 184, 294, 368);
            painter.text("Ships", centre - 184, 326, 17, menuSelection == 0 ? Gold : Ink, true);
            painter.textRight(TextFormat("%d", players), centre + 87, 326, 18, Gold, true);
            if (painter.button(painter.rect(centre + 104, 319, 34, 32), "-"))
                players = std::max(2, players - 1);
            if (painter.button(painter.rect(centre + 146, 319, 34, 32), "+"))
                players = std::min(4, players + 1);
            bots = std::clamp(bots, 0, players - 1);
            painter.text("Bots", centre - 184, 372, 17, menuSelection == 1 ? Gold : Ink, true);
            painter.textRight(TextFormat("%d", bots), centre + 87, 372, 18, Gold, true);
            if (painter.button(painter.rect(centre + 104, 365, 34, 32), "-"))
                bots = std::max(0, bots - 1);
            if (painter.button(painter.rect(centre + 146, 365, 34, 32), "+"))
                bots = std::min(players - 1, bots + 1);
            painter.text(TextFormat("%d human captain%s  /  %d automated", players - bots,
                                    players - bots == 1 ? "" : "s", bots),
                         centre - 184, 416, 13, render::theme::Dim);
            if (painter.button(painter.rect(centre - 184, 456, 368, 52), "SET SAIL", true, true) ||
                IsKeyPressed(KEY_ENTER))
                app.start(seed, players, bots);
            painter.text("Arrow controls are announced before each battle.  Esc returns here.", centre - 199,
                         568, 11, render::theme::Dim);
        } else {
            const auto model = app.model();
            auto &session = app.session();
            if (model.phase == 1 && model.activeSeat >= 0) {
                auto &client = session.client(model.activeSeat);
                const auto &state = client.state();
                if (state.build) {
                    // Commands can complete synchronously on the immediate transport and replace
                    // ClientState::build. Draw from a value snapshot so a click never invalidates
                    // the rest of this frame's layout.
                    const auto build = *state.build;
                    const auto autoFit = [&]() { autoFitClient(client); };
                    if (!handoff) {
                        const int numberKeys[]{KEY_ONE, KEY_TWO, KEY_THREE, KEY_FOUR, KEY_FIVE};
                        for (std::size_t index = 0; index < build.offer.size() && index < 5; ++index)
                            if (IsKeyPressed(numberKeys[index])) {
                                selected = build.offer[index];
                                removeMode = false;
                            }
                        if (IsKeyPressed(KEY_X))
                            removeMode = !removeMode;
                        if (IsKeyPressed(KEY_R))
                            client.command({net::RerollCommand{}});
                        if (IsKeyPressed(KEY_F))
                            client.command({net::RefitCommand{}});
                        if (IsKeyPressed(KEY_G))
                            autoFit();
                    }
                    const float centre = logicalWidth * .5f;
                    const int seconds = std::max(0, static_cast<int>(model.deadline - session.serverTime()));
                    painter.cornerTitle("BROADSIDE", TextFormat("ROUND %d  /  %s", model.round + 1,
                                                                sim::hull(build.hull).name));
                    painter.textRight(muted ? "MUTED  [M]" : "SOUND  [M]", logicalWidth - 24, 22, 11,
                                      muted ? Red : render::theme::Dim, true);
                    painter.textRight(TextFormat("%02d", seconds), logicalWidth - 24, 42, 27,
                                      seconds <= 8 ? Red : Ink, true);

                    const Rectangle offerPanel = painter.rect(18, 82, 286, 516);
                    painter.panel(offerPanel, shipColor(model.activeSeat));
                    painter.text(TextFormat("CAPTAIN %d", model.activeSeat + 1), 38, 102, 19,
                                 shipColor(model.activeSeat), true);
                    painter.text("SHIPYARD", 38, 128, 11, render::theme::Dim, true);
                    painter.textRight(TextFormat("%d", build.purse), 270, 101, 27, Gold, true);
                    painter.textRight("SCRAP", 270, 132, 9, render::theme::Dim, true);
                    painter.section("AVAILABLE PARTS", 38, 165, 246);
                    float cardY = 191;
                    for (auto partId : build.offer) {
                        const auto &part = sim::part(partId);
                        const Rectangle card = painter.rect(38, cardY, 246, 58);
                        const bool chosen = !removeMode && partId == selected;
                        DrawRectangleRec(card, chosen ? ColorAlpha(Gold, .13f) : Color{255, 255, 255, 7});
                        DrawRectangleLinesEx(card, 1, chosen ? Gold : render::theme::RuleSoft);
                        DrawRectangleRec(painter.rect(46, cardY + 9, 38, 38),
                                         render::theme::Parts[static_cast<int>(partId)]);
                        painter.text(TextFormat("%c", part.glyph), 58, cardY + 16, 17, Color{13, 18, 25, 255},
                                     true);
                        painter.textFit(part.name, 94, cardY + 8, 142, 13,
                                        build.purse >= part.cost ? Ink : ColorAlpha(render::theme::Dim, .5f),
                                        true);
                        painter.textFit(partBlurb(partId), 94, cardY + 29, 166, 8.5f, render::theme::Dim);
                        painter.textRight(TextFormat("%d", part.cost), 273, cardY + 18, 16, Gold, true);
                        if (!handoff && CheckCollisionPointRec(GetMousePosition(), card) &&
                            IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && build.purse >= part.cost) {
                            selected = partId;
                            removeMode = false;
                            audioBank.play(audio::Cue::Select, muted);
                        }
                        cardY += 64;
                    }

                    painter.text(sim::hull(build.hull).name, centre - 125, 112, 25, Ink, true);
                    painter.textFit(removeMode ? "BREAK-UP MODE - choose a fitted part"
                                               : TextFormat("Place %s - click an open deck cell",
                                                            sim::part(selected).name),
                                    centre - 125, 145, 360, 11, removeMode ? Red : render::theme::Dim);
                    const float cellSize = 43.0f;
                    const float originX = centre - cellSize * .5f;
                    const float originY = 338.0f;
                    const auto &yardHull = sim::hull(build.hull);
                    DrawEllipse(static_cast<int>(centre * uiScale), static_cast<int>(originY * uiScale),
                                (yardHull.width * cellSize * .67f) * uiScale,
                                (yardHull.length * cellSize * .46f) * uiScale,
                                ColorAlpha(shipColor(model.activeSeat), .14f));
                    DrawEllipseLines(static_cast<int>(centre * uiScale), static_cast<int>(originY * uiScale),
                                     (yardHull.width * cellSize * .67f) * uiScale,
                                     (yardHull.length * cellSize * .46f) * uiScale,
                                     ColorAlpha(shipColor(model.activeSeat), .45f));
                    DrawTriangle(
                        {(centre - 21) * uiScale, (originY + yardHull.bowZ * cellSize * .74f - 5) * uiScale},
                        {centre * uiScale, (originY + yardHull.bowZ * cellSize * .74f - 36) * uiScale},
                        {(centre + 21) * uiScale, (originY + yardHull.bowZ * cellSize * .74f - 5) * uiScale},
                        ColorAlpha(shipColor(model.activeSeat), .55f));
                    for (auto cell : yardHull.cells) {
                        Rectangle box =
                            painter.rect(originX + cell.dx * cellSize, originY + cell.dz * cellSize * .74f,
                                         cellSize - 4, cellSize * .70f);
                        bool hover = CheckCollisionPointRec(GetMousePosition(), box);
                        const Color deck = cell.dx == 0 ? Color{51, 70, 84, 255} : Color{37, 51, 61, 255};
                        DrawRectangleRec(box, hover ? Color{68, 95, 92, 255} : deck);
                        DrawRectangleLinesEx(box, hover ? 2 : 1, hover ? Gold : Color{95, 168, 255, 90});
                        auto it = build.design.parts.find(cell);
                        if (it != build.design.parts.end()) {
                            DrawRectangleRec(
                                {box.x + 4 * uiScale, box.y + 4 * uiScale, box.width - 8 * uiScale,
                                 box.height - 8 * uiScale},
                                ColorAlpha(render::theme::Parts[static_cast<int>(it->second.id)],
                                           it->second.hp < sim::part(it->second.id).hp ? .55f : 1.0f));
                            const char glyph[2]{sim::part(it->second.id).glyph, 0};
                            DrawTextEx(renderer.headingFont(), glyph,
                                       {box.x + box.width * .36f, box.y + box.height * .16f},
                                       box.height * .56f, .3f, Color{12, 17, 20, 240});
                        } else if (hover) {
                            const auto &ghost = sim::part(selected);
                            DrawRectangleRec(
                                {box.x + 4 * uiScale, box.y + 4 * uiScale, box.width - 8 * uiScale,
                                 box.height - 8 * uiScale},
                                ColorAlpha(render::theme::Parts[static_cast<int>(selected)], .48f));
                            if (ghost.gun) {
                                Vector2 center{box.x + box.width / 2, box.y + box.height / 2};
                                float aim = ghost.gunSpec.arc == sim::Arc::Bow
                                                ? -90.0f
                                                : (cell.dx < 0 ? 180.0f : 0.0f);
                                DrawCircleSectorLines(center, cellSize * 4,
                                                      aim - static_cast<float>(ghost.gunSpec.halfArc),
                                                      aim + static_cast<float>(ghost.gunSpec.halfArc), 24,
                                                      ColorAlpha(Gold, .65f));
                            }
                        }
                        if (!handoff && hover && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
                            if (removeMode && it != build.design.parts.end())
                                client.command({net::RemoveCommand{cell}});
                            else if (!removeMode && it == build.design.parts.end())
                                client.command({net::PlaceCommand{cell, selected}});
                        }
                        if (!handoff && hover && IsMouseButtonPressed(MOUSE_BUTTON_RIGHT) &&
                            it != build.design.parts.end())
                            client.command({net::RemoveCommand{cell}});
                    }

                    game::Shipyard readout(build.design, build.hull, build.purse);
                    const auto &stats = readout.stats();
                    const Rectangle statsPanel = painter.rect(logicalWidth - 272, 82, 254, 516);
                    painter.panel(statsPanel);
                    painter.section("SHIP READOUT", logicalWidth - 250, 105, 210);
                    const std::array<std::pair<const char *, std::string>, 5> rows{
                        {{"Open holes", std::to_string(stats.total - stats.used)},
                         {"Guns", std::to_string(stats.guns)},
                         {"Crew",
                          std::to_string(stats.crewSupply) + " of " + std::to_string(stats.crewNeeded)},
                         {"Masts", std::to_string(stats.masts) + " of " + std::to_string(stats.mastsWanted)},
                         {"Powder", std::to_string(stats.magazines)}}};
                    float rowY = 139;
                    for (const auto &[label, value] : rows) {
                        painter.text(label, logicalWidth - 250, rowY, 11, render::theme::Dim);
                        painter.textRight(value, logicalWidth - 40, rowY - 2, 14, Ink, true);
                        rowY += 31;
                    }
                    painter.section("WARNINGS", logicalWidth - 250, 305, 210);
                    rowY = 334;
                    for (const auto &warning : build.warnings) {
                        DrawRectangleRec(painter.rect(logicalWidth - 250, rowY - 5, 210, 44),
                                         ColorAlpha(Red, .12f));
                        DrawRectangleRec(painter.rect(logicalWidth - 250, rowY - 5, 2, 44), Red);
                        painter.textWrapped(warning, logicalWidth - 240, rowY, 190, 9, 16, 2,
                                            Color{255, 187, 168, 255});
                        rowY += 50;
                    }
                    if (state.lastResult) {
                        rowY = std::max(rowY + 5, 470.0f);
                        painter.section("LAST SEEN", logicalWidth - 250, rowY, 210);
                        rowY += 24;
                        for (const auto &ship : state.lastResult->ships)
                            if (ship.seat != model.activeSeat) {
                                painter.text(TextFormat("C%d", ship.seat + 1), logicalWidth - 250, rowY, 11,
                                             shipColor(ship.seat), true);
                                painter.progress(painter.rect(logicalWidth - 218, rowY + 2, 70, 8),
                                                 ship.soundness / 100.0f);
                                painter.text(TextFormat("%d guns  %d hands", ship.firingGuns, ship.hands),
                                             logicalWidth - 136, rowY, 9, render::theme::Dim);
                                rowY += 22;
                            }
                    }
                    if (!state.lastDenial.empty()) {
                        DrawRectangleRec(painter.rect(centre - 190, logicalHeight - 116, 380, 30),
                                         ColorAlpha(Red, .20f));
                        painter.textFit(state.lastDenial, centre - 178, logicalHeight - 109, 356, 10, Red);
                    }
                    if ((painter.button(painter.rect(centre - 126, logicalHeight - 68, 252, 44),
                                        "LOCK IN  [SPACE]", !handoff, true) ||
                         (!handoff && IsKeyPressed(KEY_SPACE))))
                        client.command({net::LockCommand{}});
                    if (painter.button(painter.rect(38, 531, 76, 43), "REROLL", !handoff && build.purse >= 2))
                        client.command({net::RerollCommand{}});
                    if (painter.button(painter.rect(122, 531, 76, 43), "REFIT", !handoff))
                        client.command({net::RefitCommand{}});
                    if (painter.button(painter.rect(206, 531, 78, 43), "AUTO", !handoff))
                        autoFit();
                    if (painter.button(painter.rect(logicalWidth - 250, 531, 210, 43), "BREAK UP", !handoff,
                                       removeMode, Red))
                        removeMode = !removeMode;
                }
            } else if (model.phase == 2) {
                const auto *battle = session.battle();
                if (battle) {
                    DrawRectangleRec(painter.rect(0, 0, logicalWidth, 62), Color{7, 12, 18, 224});
                    DrawLineEx({0, 62 * uiScale}, {static_cast<float>(width), 62 * uiScale}, 1,
                               render::theme::Rule);
                    std::string score;
                    const auto &scores = session.state().scores;
                    for (const auto &ship : battle->state()) {
                        if (!score.empty())
                            score += "   ";
                        score += "C" + std::to_string(ship.index + 1) + "  " +
                                 std::to_string(ship.index < static_cast<int>(scores.size())
                                                    ? scores[static_cast<std::size_t>(ship.index)]
                                                    : 0);
                    }
                    painter.text(score, 18, 17, 15, Ink, true);
                    painter.text(TextFormat("ROUND %d", model.round + 1), 300, 10, 11, Ink, true);
                    painter.text(sim::hull(battle->hullIndex()).name, 300, 31, 9, render::theme::Dim, true);
                    const float windX = logicalWidth * .5f;
                    painter.compass({windX * uiScale, 30 * uiScale}, 23 * uiScale, battle->windTo());
                    painter.text(windName(battle->windTo()), windX + 34, 21, 9, render::theme::Dim, true);
                    painter.textRight(muted ? "MUTED  [M]" : "SOUND  [M]", logicalWidth - 87, 18, 10,
                                      muted ? Red : render::theme::Dim, true);
                    painter.textRight(TextFormat("%02d", std::max(0, static_cast<int>(40 - battle->time()))),
                                      logicalWidth - 17, 10, 27, battle->time() > 32 ? Red : Ink, true);

                    std::array<int, 2> sideRows{};
                    const int ammoKeys[] = {KEY_A, KEY_L, KEY_Q, KEY_P};
                    for (const auto &ship : battle->state()) {
                        const int sideIndex = ship.index % 2;
                        const int row = sideRows[static_cast<std::size_t>(sideIndex)]++;
                        const float panelX = sideIndex == 0 ? 14.0f : logicalWidth - 232.0f;
                        const float panelY = 78.0f + row * 146.0f;
                        const bool human = ship.index < session.humans();
                        const float panelHeight = human ? 132.0f : 91.0f;
                        const Rectangle panel = painter.rect(panelX, panelY, 218, panelHeight);
                        painter.panel(panel, shipColor(ship.index), sideIndex == 1);
                        painter.text(TextFormat("CAPTAIN %d%s", ship.index + 1,
                                                ship.index >= session.humans() ? "  /  BOT" : ""),
                                     panelX + 14, panelY + 12, 10, shipColor(ship.index), true);
                        const float structure = static_cast<float>(battle->structureFraction(ship));
                        painter.progress(painter.rect(panelX + 14, panelY + 34, 190, 9), structure);
                        int liveGuns = 0, mannedGuns = 0;
                        for (const auto &gun : ship.guns) {
                            const bool alive = ship.cells[static_cast<std::size_t>(gun.cell)].alive;
                            liveGuns += alive;
                            mannedGuns += alive && gun.manned;
                        }
                        painter.text(ship.out ? "STRUCK"
                                              : TextFormat("hands  %d / %d", ship.crew, ship.crewSupply),
                                     panelX + 14, panelY + 53, 9, ship.out ? Red : render::theme::Dim, true);
                        painter.textRight(ship.magazines == 0
                                              ? "NO POWDER"
                                              : TextFormat("guns  %d / %d", mannedGuns, liveGuns),
                                          panelX + 204, panelY + 53, 9,
                                          ship.magazines == 0 ? Red : render::theme::Dim, true);
                        if (human) {
                            const Rectangle roundButton = painter.rect(panelX + 14, panelY + 80, 91, 30);
                            const Rectangle grapeButton = painter.rect(panelX + 113, panelY + 80, 91, 30);
                            if (painter.button(roundButton, "ROUND", !ship.out, ship.ammo == sim::Ammo::Round,
                                               Gold))
                                session.client(ship.index).command({net::SetAmmoCommand{sim::Ammo::Round}});
                            if (painter.button(grapeButton, "GRAPE", !ship.out, ship.ammo == sim::Ammo::Grape,
                                               Gold))
                                session.client(ship.index).command({net::SetAmmoCommand{sim::Ammo::Grape}});
                            painter.text(TextFormat("KEY  %c", "ALQP"[ship.index]), panelX + 83, panelY + 116,
                                         8, render::theme::Dim, true);
                            if (IsKeyPressed(ammoKeys[ship.index]))
                                session.client(ship.index)
                                    .command({net::SetAmmoCommand{ship.ammo == sim::Ammo::Round
                                                                      ? sim::Ammo::Grape
                                                                      : sim::Ammo::Round}});
                        }
                    }

                    const auto &log = battle->log();
                    float logY = logicalHeight - 38;
                    for (std::size_t i = log.size() > 3 ? log.size() - 3 : 0; i < log.size(); ++i) {
                        const std::string signal =
                            TextFormat("C%d  %s", log[i].ship + 1, log[i].text.c_str());
                        const float textWidth = painter.measure(signal, 10, true) / uiScale;
                        DrawRectangleRec(painter.rect(logicalWidth * .5f - textWidth * .5f - 12, logY - 4,
                                                      textWidth + 24, 23),
                                         Color{8, 12, 18, 215});
                        DrawRectangleRec(
                            painter.rect(logicalWidth * .5f - textWidth * .5f - 12, logY - 4, 2, 23), Gold);
                        painter.text(signal, logicalWidth * .5f - textWidth * .5f, logY, 10, Ink, true);
                        logY -= 27;
                    }
                    if (debugOverlay) {
                        const auto &rs = renderer.stats();
                        DrawRectangleRec(painter.rect(logicalWidth - 260, logicalHeight - 76, 244, 54),
                                         Color{2, 5, 8, 210});
                        painter.text(
                            TextFormat("tick %d  checksum %08x", battle->tickCount(), sim::checksum(*battle)),
                            logicalWidth - 248, logicalHeight - 66, 9, render::theme::Dim);
                        painter.text(TextFormat("render %.0f%%  fx %d/720  overflow %d", rs.renderScale * 100,
                                                rs.particles, rs.particleOverflow),
                                     logicalWidth - 248, logicalHeight - 45, 9, render::theme::Dim);
                    }
                }
            } else if (model.phase == 3 || model.phase == 4) {
                const auto &result = session.state().result;
                const float centre = logicalWidth * .5f;
                const Rectangle resultPanel = painter.rect(centre - 390, 70, 780, 568);
                painter.panel(resultPanel);
                painter.text(model.phase == 4 ? "MATCH COMPLETE" : "ROUND COMPLETE", centre - 350, 99, 13,
                             Gold, true);
                if (result) {
                    const int victor = model.phase == 4 ? result->matchWinner : result->winner;
                    const std::string verdict =
                        victor < 0 ? "HONOURS EVEN" : TextFormat("CAPTAIN %d TAKES THE DAY", victor + 1);
                    painter.text(verdict, centre - painter.measure(verdict, 34, true) / uiScale * .5f, 130,
                                 34, victor < 0 ? Ink : shipColor(victor), true);
                    painter.text(result->reason, centre - painter.measure(result->reason, 12) / uiScale * .5f,
                                 177, 12, render::theme::Dim);
                    const float scoreWidth = result->scores.size() * 98.0f;
                    float scoreX = centre - scoreWidth * .5f;
                    for (std::size_t seat = 0; seat < result->scores.size(); ++seat) {
                        painter.text(TextFormat("C%d", static_cast<int>(seat) + 1), scoreX, 210, 10,
                                     shipColor(static_cast<int>(seat)), true);
                        painter.text(TextFormat("%d", result->scores[seat]), scoreX + 32, 202, 24, Ink, true);
                        scoreX += 98;
                    }
                    painter.section("THE FLEET", centre - 350, 248, 700);
                    float y = 278;
                    for (const auto &ship : result->ships) {
                        DrawRectangleRec(painter.rect(centre - 350, y - 6, 700, 48), Color{255, 255, 255, 7});
                        DrawRectangleRec(painter.rect(centre - 350, y - 6, 3, 48), shipColor(ship.seat));
                        painter.text(TextFormat("CAPTAIN %d", ship.seat + 1), centre - 334, y + 2, 11,
                                     shipColor(ship.seat), true);
                        painter.progress(painter.rect(centre - 218, y + 5, 125, 9), ship.soundness / 100.0f);
                        painter.text(TextFormat("%d%% sound", ship.soundness), centre - 82, y + 1, 10,
                                     render::theme::Dim);
                        painter.text(TextFormat("%d firing guns", ship.firingGuns), centre + 27, y + 1, 10,
                                     Ink);
                        painter.text(TextFormat("%d hands", ship.hands), centre + 145, y + 1, 10, Ink);
                        painter.text(TextFormat("%d powder", ship.powder), centre + 227, y + 1, 10, Ink);
                        painter.textRight(TextFormat("%d masts lost", ship.lostMasts), centre + 334, y + 22,
                                          9, ship.lostMasts ? Red : render::theme::Dim);
                        y += 55;
                    }
                    y += 5;
                    painter.section("LAST SIGNALS", centre - 350, y, 700);
                    y += 27;
                    for (std::size_t i = result->log.size() > 4 ? result->log.size() - 4 : 0;
                         i < result->log.size(); ++i) {
                        painter.text(TextFormat("%04.1f", result->log[i].time), centre - 350, y, 9,
                                     render::theme::Dim, true);
                        painter.text(TextFormat("C%d", result->log[i].ship + 1), centre - 303, y, 9,
                                     shipColor(result->log[i].ship), true);
                        painter.textFit(result->log[i].text, centre - 272, y, 610, 10, Ink);
                        y += 20;
                    }
                }
                if (painter.button(painter.rect(centre - 136, 573, 272, 42),
                                   model.phase == 4 ? "REMATCH  [ENTER]" : "CONTINUE  [ENTER]", true, true) ||
                    IsKeyPressed(KEY_ENTER)) {
                    for (int seat = 0; seat < session.humans(); ++seat)
                        session.client(seat).command({model.phase == 4
                                                          ? net::CommandPayload{net::RematchCommand{}}
                                                          : net::CommandPayload{net::ContinueCommand{}}});
                }
            }
            if (handoff && model.phase == 1) {
                DrawRectangle(0, 0, width, height, Color{5, 9, 14, 247});
                const float centre = logicalWidth * .5f;
                painter.panel(painter.rect(centre - 260, 190, 520, 300), shipColor(model.activeSeat));
                painter.text("CAPTAIN'S HAND", centre - 75, 226, 11, Gold, true);
                const std::string captain = TextFormat("CAPTAIN %d", model.activeSeat + 1);
                painter.text(captain, centre - painter.measure(captain, 36, true) / uiScale * .5f, 263, 36,
                             shipColor(model.activeSeat), true);
                painter.textWrapped(
                    "Pass the controls. The next shipyard stays private until its captain is ready.",
                    centre - 205, 322, 410, 12, 19, 2, render::theme::Dim);
                if (painter.button(painter.rect(centre - 170, 382, 340, 48), "TAKE THE HELM  [ENTER]", true,
                                   true) ||
                    IsKeyPressed(KEY_ENTER))
                    handoff = false;
            }
        }
        bool captured = false;
        if (!capturePath.empty() && captureWarmupFrames-- <= 0) {
            rlDrawRenderBatchActive();
            Image image = LoadImageFromScreen();
            std::filesystem::path output(capturePath);
            if (output.has_parent_path())
                std::filesystem::create_directories(output.parent_path());
            ExportImage(image, capturePath.c_str());
            Color *pixels = LoadImageColors(image);
            double sum = 0, sumSq = 0;
            int count = image.width * image.height;
            for (int i = 0; i < count; ++i) {
                double l = .2126 * pixels[i].r + .7152 * pixels[i].g + .0722 * pixels[i].b;
                sum += l;
                sumSq += l * l;
            }
            double mean = sum / count, variance = sumSq / count - mean * mean;
            std::uint64_t hash = 0;
            for (int by = 0; by < 8; ++by)
                for (int bx = 0; bx < 8; ++bx) {
                    const Color p = pixels[(by * image.height / 8) * image.width + bx * image.width / 8];
                    double l = .2126 * p.r + .7152 * p.g + .0722 * p.b;
                    hash = (hash << 1) | (l >= mean);
                }
            UnloadImageColors(pixels);
            bool valid = mean > 5 && mean < 245 && variance > 80;
            if (scenario == "duel")
                valid = valid && renderer.stats().ships == 2 && renderer.stats().particles > 0 &&
                        renderer.stats().flags == 2 && renderer.stats().particleOverflow == 0;
            if (scenario == "sinking")
                valid = valid && renderer.stats().ships == 2 && renderer.stats().sinkingShips == 1 &&
                        renderer.stats().flags >= 1 && renderer.stats().particleOverflow == 0;
            if (scenario == "four-way")
                valid = valid && renderer.stats().ships == 4 && renderer.stats().particles > 0 &&
                        renderer.stats().flags == 4 && renderer.stats().particleOverflow == 0;
            if (!baselinePath.empty()) {
                std::ifstream baselines(baselinePath);
                std::string name, hex;
                int bw = 0, bh = 0;
                bool found = false;
                while (baselines >> name >> bw >> bh >> hex)
                    if (name == scenario && bw == image.width && bh == image.height) {
                        std::uint64_t expected = std::stoull(hex, nullptr, 16);
                        valid = valid && std::popcount(hash ^ expected) <= 10;
                        found = true;
                        break;
                    }
                valid = valid && found;
            }
            std::cout << "capture: " << (scenario.empty() ? "live" : scenario) << ' ' << image.width << 'x'
                      << image.height << " mean=" << mean << " variance=" << variance << " ahash=" << std::hex
                      << hash << std::dec << " ships=" << renderer.stats().ships
                      << " parts=" << renderer.stats().parts << " effects=" << renderer.stats().particles
                      << " flags=" << renderer.stats().flags << " sinking=" << renderer.stats().sinkingShips
                      << " overflow=" << renderer.stats().particleOverflow << (valid ? " PASS" : " FAIL")
                      << '\n';
            if (!valid)
                exitCode = 1;
            UnloadImage(image);
            captured = true;
        }
        EndDrawing();
        if (captured)
            break;
    }
    renderer.unload();
    CloseWindow();
    return exitCode;
}
#else
#include <iostream>
int main() {
    std::cerr << "Broadside was built without raylib. Configure with -DBROADSIDE_DESKTOP=ON.\n";
    return 0;
}
#endif
