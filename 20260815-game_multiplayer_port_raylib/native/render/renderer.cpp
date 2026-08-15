#include "renderer.h"

#include "raymath.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <string>

namespace broadside::render {
namespace {

constexpr Color SeaDeep{16, 41, 52, 255};
constexpr Color Foam{166, 205, 209, 220};
constexpr std::array<Color, 4> PlayerColors{
    {{95, 168, 255, 255}, {255, 122, 95, 255}, {99, 209, 168, 255}, {201, 140, 240, 255}}};
constexpr std::array<Color, 4> DeckColors{
    {{37, 51, 61, 255}, {61, 42, 36, 255}, {36, 58, 49, 255}, {51, 40, 61, 255}}};
constexpr std::array<Color, 4> SpineColors{
    {{51, 70, 84, 255}, {81, 58, 48, 255}, {48, 80, 63, 255}, {69, 54, 85, 255}}};
constexpr std::array<Color, 4> HullColors{
    {{61, 85, 104, 255}, {96, 64, 58, 255}, {58, 96, 77, 255}, {84, 63, 102, 255}}};
constexpr std::array<Color, 10> PartColors{{{138, 104, 68, 255},
                                            {77, 90, 94, 255},
                                            {201, 162, 39, 255},
                                            {216, 203, 176, 255},
                                            {176, 48, 48, 255},
                                            {127, 168, 201, 255},
                                            {74, 111, 165, 255},
                                            {139, 95, 176, 255},
                                            {47, 143, 111, 255},
                                            {232, 232, 232, 255}}};

constexpr const char *SeaFragment = R"glsl(
#version 330
in vec2 fragTexCoord;
in vec4 fragColor;
out vec4 finalColor;
uniform vec2 resolution;
uniform float time;
uniform vec2 wind;
uniform vec4 worldMap;
uniform float worldPixel;
uniform vec3 arenaRing;
const float TAU = 6.28318530718;
void wave(vec2 p, vec2 direction, float wavelength, float steepness, float speed,
          inout float height, inout vec2 slope, inout vec3 phases) {
    float k = TAU / wavelength;
    float phase = k * dot(direction, p) - speed * time;
    float s = sin(phase);
    float c = cos(phase);
    height += (steepness / k) * s;
    slope += direction * steepness * c;
    phases = vec3(phases.yz, s);
}
void main() {
    vec2 w = normalize(wind + vec2(.0001));
    vec2 uv = gl_FragCoord.xy / max(resolution, vec2(1.0));
    vec2 p = worldMap.xy + (uv - .5) * worldMap.zw;
    vec2 across = vec2(-w.y, w.x);
    vec2 d1 = w * .547786 + across * .836619;
    vec2 d2 = w * .681604 - across * .731722;
    float warpA = sin(dot(p, vec2(.031, -.019)) + time * .07);
    float warpB = sin(dot(p, vec2(-.017, .027)) - time * .05);
    vec2 samplePoint = p + 3.6 * vec2(warpA, warpB);
    float broad = .5 + .25 * (warpA + warpB);
    float height = 0.0;
    vec2 slope = vec2(0.0);
    vec3 phases = vec3(0.0);
    wave(samplePoint, w, 16.0, .11, 1.15, height, slope, phases);
    wave(samplePoint, d1, 10.5, .10, 1.50, height, slope, phases);
    wave(samplePoint, d2, 7.2, .07, 1.90, height, slope, phases);
    float waveHeight = height / .53;
    vec3 normal = normalize(vec3(-slope.x, 1.0, -slope.y));
    float light = dot(normal, normalize(vec3(-.42, .88, .22)));
    vec3 deep = vec3(.063, .161, .204);
    vec3 water = vec3(.090, .216, .259);
    vec3 swell = vec3(.129, .298, .353);
    vec3 glint = vec3(.275, .467, .510);
    vec3 foam = vec3(.651, .804, .820);
    float body = clamp(.47 + broad * .10 + waveHeight * .035, 0.0, 1.0);
    vec3 color = mix(deep, water, body);
    color = mix(color, swell, smoothstep(.22, .72, waveHeight) * .09);
    color = mix(color, glint, smoothstep(.91, .985, light) * .06);
    float crest = waveHeight + length(slope) * .18;
    float breakup = .5 + .25 * sin(dot(p, across) * .61 + phases.z * 1.3 - time * .18)
        + .25 * sin(dot(p, d1) * .37 - phases.x * 1.7 + time * .11);
    float cap = smoothstep(.70, .84, crest) * (1.0 - smoothstep(.90, 1.02, crest));
    float detail = clamp((.55 - worldPixel) / .35, 0.0, 1.0);
    color = mix(color, foam, cap * smoothstep(.64, .94, breakup) * detail * .18);
    float etched = fract((uv.x * 1.15 + uv.y * .55) * 24.0
                         + dot(worldMap.xy, d1) * .018 - time * .025);
    float etchedLine = smoothstep(.965, .985, etched);
    float brokenStreak = smoothstep(.45, .78, .5 + .5 * sin((uv.x - uv.y) * 78.0
                                                            + warpB * 2.4));
    color = mix(color, glint, etchedLine * brokenStreak * detail * .28);
    float ringDelta = abs(dot(p, p) - arenaRing.x);
    float ring = 1.0 - smoothstep(arenaRing.y, arenaRing.y + worldPixel * 1.5 * arenaRing.z, ringDelta);
    color = mix(color, vec3(.204, .439, .514), ring * .38);
    finalColor = vec4(color, 1.0);
})glsl";

constexpr const char *PostFragment = R"glsl(
#version 330
in vec2 fragTexCoord;
in vec4 fragColor;
out vec4 finalColor;
uniform sampler2D texture0;
uniform vec2 inverseResolution;
float luma(vec3 rgb) { return dot(rgb, vec3(.299, .587, .114)); }
void main() {
    vec2 uv = fragTexCoord;
    vec3 nw = texture(texture0, uv + vec2(-1.0, -1.0) * inverseResolution).rgb;
    vec3 ne = texture(texture0, uv + vec2( 1.0, -1.0) * inverseResolution).rgb;
    vec3 sw = texture(texture0, uv + vec2(-1.0,  1.0) * inverseResolution).rgb;
    vec3 se = texture(texture0, uv + vec2( 1.0,  1.0) * inverseResolution).rgb;
    vec3 mid = texture(texture0, uv).rgb;
    float lnw = luma(nw), lne = luma(ne), lsw = luma(sw), lse = luma(se), lm = luma(mid);
    vec2 dir = vec2(-((lnw + lne) - (lsw + lse)), (lnw + lsw) - (lne + lse));
    float reduce = max((lnw + lne + lsw + lse) * (.25 * .125), 1.0 / 128.0);
    float reciprocal = 1.0 / (min(abs(dir.x), abs(dir.y)) + reduce);
    dir = clamp(dir * reciprocal, vec2(-8.0), vec2(8.0)) * inverseResolution;
    vec3 a = .5 * (texture(texture0, uv + dir * (1.0 / 3.0 - .5)).rgb
                 + texture(texture0, uv + dir * (2.0 / 3.0 - .5)).rgb);
    vec3 b = a * .5 + .25 * (texture(texture0, uv + dir * -.5).rgb
                            + texture(texture0, uv + dir * .5).rgb);
    float low = min(lm, min(min(lnw, lne), min(lsw, lse)));
    float high = max(lm, max(max(lnw, lne), max(lsw, lse)));
    float lb = luma(b);
    finalColor = vec4((lb < low || lb > high) ? a : b, 1.0) * fragColor;
})glsl";

std::filesystem::path assetPath(const char *relative) {
    const auto beside = std::filesystem::path(GetApplicationDirectory()) / "assets" / relative;
    if (std::filesystem::exists(beside))
        return beside;
#ifdef BROADSIDE_ASSET_DIR
    return std::filesystem::path(BROADSIDE_ASSET_DIR) / relative;
#else
    return beside;
#endif
}

Matrix transform(float x, float y, float z, float sx, float sy, float sz, float heading) {
    Matrix matrix = MatrixMultiply(MatrixScale(sx, sy, sz), MatrixRotateY(heading));
    matrix = MatrixMultiply(matrix, MatrixTranslate(x, y, z));
    return matrix;
}

Vector3 worldCell(const sim::RuntimeShip &ship, const sim::Coord &coord, float y) {
    const float side = static_cast<float>(coord.dx * sim::CellSize);
    const float fore = static_cast<float>(coord.dz * sim::CellSize);
    return {static_cast<float>(ship.position.x) + static_cast<float>(ship.cosHeading) * side -
                static_cast<float>(ship.sinHeading) * fore,
            y,
            static_cast<float>(ship.position.z) + static_cast<float>(ship.sinHeading) * side +
                static_cast<float>(ship.cosHeading) * fore};
}

void drawEllipse(Vector3 center, float wide, float lon, float heading, Color color) {
    Vector3 previous{};
    for (int i = 0; i <= 40; ++i) {
        const float angle = i * (6.2831853f / 40.0f);
        const float side = std::cos(angle) * wide;
        const float fore = std::sin(angle) * lon;
        const Vector3 point{center.x + std::cos(heading) * side + std::sin(heading) * fore, center.y,
                            center.z + std::sin(heading) * side - std::cos(heading) * fore};
        if (i)
            DrawLine3D(previous, point, color);
        previous = point;
    }
}

} // namespace

DesktopRenderer::DesktopRenderer() {
    seaShader_ = LoadShaderFromMemory(nullptr, SeaFragment);
    postShader_ = LoadShaderFromMemory(nullptr, PostFragment);
    resolutionLocation_ = GetShaderLocation(seaShader_, "resolution");
    timeLocation_ = GetShaderLocation(seaShader_, "time");
    windLocation_ = GetShaderLocation(seaShader_, "wind");
    mapLocation_ = GetShaderLocation(seaShader_, "worldMap");
    pixelLocation_ = GetShaderLocation(seaShader_, "worldPixel");
    ringLocation_ = GetShaderLocation(seaShader_, "arenaRing");
    inverseResolutionLocation_ = GetShaderLocation(postShader_, "inverseResolution");
    cube_ = GenMeshCube(1.0f, 1.0f, 1.0f);
    for (std::size_t i = 0; i < partMaterials_.size(); ++i) {
        partMaterials_[i] = LoadMaterialDefault();
        partMaterials_[i].maps[MATERIAL_MAP_DIFFUSE].color = ColorBrightness(PartColors[i], .10f);
    }
    for (std::size_t i = 0; i < hullMaterials_.size(); ++i) {
        hullMaterials_[i] = LoadMaterialDefault();
        hullMaterials_[i].maps[MATERIAL_MAP_DIFFUSE].color = ColorBrightness(HullColors[i], -.24f);
    }
    const auto headingPath = assetPath("fonts/IMFeENsc28P.ttf").string();
    const auto bodyPath = assetPath("fonts/Inter-Variable.ttf").string();
    heading_ = LoadFontEx(headingPath.c_str(), 72, nullptr, 0);
    body_ = LoadFontEx(bodyPath.c_str(), 40, nullptr, 0);
    SetTextureFilter(heading_.texture, TEXTURE_FILTER_BILINEAR);
    SetTextureFilter(body_.texture, TEXTURE_FILTER_BILINEAR);
}

DesktopRenderer::~DesktopRenderer() {
    unload();
}

void DesktopRenderer::unload() {
    if (!loaded_)
        return;
    loaded_ = false;
    if (target_.id)
        UnloadRenderTexture(target_);
    if (heading_.texture.id)
        UnloadFont(heading_);
    if (body_.texture.id)
        UnloadFont(body_);
    for (auto &material : partMaterials_)
        UnloadMaterial(material);
    for (auto &material : hullMaterials_)
        UnloadMaterial(material);
    UnloadMesh(cube_);
    if (seaShader_.id)
        UnloadShader(seaShader_);
    if (postShader_.id)
        UnloadShader(postShader_);
}

void DesktopRenderer::ensureTarget(int width, int height) {
    const float scale = quality_.scale();
    constexpr float FillBudget = 1920.0f * 1080.0f;
    const float pixels = static_cast<float>(std::max(1, width) * std::max(1, height));
    const float supersample = pixels <= FillBudget ? std::min(1.50f, std::sqrt(FillBudget / pixels)) : 1.0f;
    const int wantedWidth = std::max(1, static_cast<int>(std::round(width * scale * supersample)));
    const int wantedHeight = std::max(1, static_cast<int>(std::round(height * scale * supersample)));
    if (target_.id && wantedWidth == targetWidth_ && wantedHeight == targetHeight_)
        return;
    if (target_.id)
        UnloadRenderTexture(target_);
    target_ = LoadRenderTexture(wantedWidth, wantedHeight);
    SetTextureFilter(target_.texture, TEXTURE_FILTER_BILINEAR);
    targetWidth_ = wantedWidth;
    targetHeight_ = wantedHeight;
}

void DesktopRenderer::drawSeaIntoTarget(float time, double windTo, Vector3 centre, float viewHeight,
                                        float arenaRadius) {
    const float resolution[2]{static_cast<float>(targetWidth_), static_cast<float>(targetHeight_)};
    const float wind[2]{static_cast<float>(std::sin(windTo)), static_cast<float>(-std::cos(windTo))};
    const float aspect = targetWidth_ / static_cast<float>(std::max(1, targetHeight_));
    const float worldMap[4]{centre.x, centre.z, viewHeight * aspect, -viewHeight / std::sin(60.0f * DEG2RAD)};
    const float worldPixel = viewHeight * aspect / std::max(1, targetWidth_);
    const float ringCentre = arenaRadius - .45f;
    const float ring[3]{ringCentre * ringCentre, 2 * ringCentre * .45f, 2 * ringCentre};
    SetShaderValue(seaShader_, resolutionLocation_, resolution, SHADER_UNIFORM_VEC2);
    SetShaderValue(seaShader_, timeLocation_, &time, SHADER_UNIFORM_FLOAT);
    SetShaderValue(seaShader_, windLocation_, wind, SHADER_UNIFORM_VEC2);
    SetShaderValue(seaShader_, mapLocation_, worldMap, SHADER_UNIFORM_VEC4);
    SetShaderValue(seaShader_, pixelLocation_, &worldPixel, SHADER_UNIFORM_FLOAT);
    SetShaderValue(seaShader_, ringLocation_, ring, SHADER_UNIFORM_VEC3);
    BeginShaderMode(seaShader_);
    DrawRectangle(0, 0, targetWidth_, targetHeight_, WHITE);
    EndShaderMode();
}

void DesktopRenderer::drawSea(int screenWidth, int screenHeight, float time, double windTo) {
    ensureTarget(screenWidth, screenHeight);
    BeginTextureMode(target_);
    ClearBackground(SeaDeep);
    drawSeaIntoTarget(time, windTo, {}, 120.0f, 60.0f);
    EndTextureMode();
    presentTarget(screenWidth, screenHeight);
}

void DesktopRenderer::presentTarget(int screenWidth, int screenHeight) {
    const float inverse[2]{1.0f / std::max(1, targetWidth_), 1.0f / std::max(1, targetHeight_)};
    SetShaderValue(postShader_, inverseResolutionLocation_, inverse, SHADER_UNIFORM_VEC2);
    BeginShaderMode(postShader_);
    DrawTexturePro(
        target_.texture,
        {0, 0, static_cast<float>(target_.texture.width), -static_cast<float>(target_.texture.height)},
        {0, 0, static_cast<float>(screenWidth), static_cast<float>(screenHeight)}, {}, 0, WHITE);
    EndShaderMode();
}

void DesktopRenderer::resetEffects() {
    for (auto &particle : particles_)
        particle.active = false;
    effectCursor_ = 0;
    effectSeed_ = 0;
    nextParticle_ = 0;
    stats_.particleOverflow = 0;
}

void DesktopRenderer::spawn(const sim::Effect &effect, int count, Color color, float speed, float life,
                            float size) {
    for (int i = 0; i < count; ++i) {
        Particle *slot = nullptr;
        for (std::size_t probe = 0; probe < particles_.size(); ++probe) {
            auto &candidate = particles_[(nextParticle_ + probe) % particles_.size()];
            if (!candidate.active) {
                slot = &candidate;
                nextParticle_ = (nextParticle_ + probe + 1) % particles_.size();
                break;
            }
        }
        if (!slot) {
            ++stats_.particleOverflow;
            continue;
        }
        const std::uint32_t hash =
            static_cast<std::uint32_t>((effectCursor_ + 1) * 747796405u + i * 2891336453u);
        const float angle = static_cast<float>(hash & 0xffffu) * (6.2831853f / 65536.0f);
        const float radial = speed * (0.35f + static_cast<float>((hash >> 16) & 255u) / 255.0f);
        slot->position = {static_cast<float>(effect.pos.x), 1.0f, static_cast<float>(effect.pos.z)};
        slot->velocity = {std::sin(angle) * radial, speed * (.35f + (i % 5) * .13f),
                          std::cos(angle) * radial};
        slot->age = 0;
        slot->life = life * (.8f + (i % 4) * .1f);
        slot->size = size * (.75f + (i % 3) * .18f);
        slot->color = color;
        slot->active = true;
        slot->mast = effect.part == sim::PartId::Mast && i == 0;
    }
}

void DesktopRenderer::consumeEffects(const sim::Battle &battle) {
    if (effectSeed_ != battle.seed() || effectCursor_ > battle.effects().size()) {
        resetEffects();
        effectSeed_ = battle.seed();
        // A reconnect or deterministic capture may attach after several seconds of battle. Only
        // recent signals can still be visible; replaying the whole historical stream would fill
        // the fixed pool with already-expired smoke on a single frame.
        if (battle.effects().size() > 36)
            effectCursor_ = battle.effects().size() - 36;
    }
    const int density = std::max(1, 5 - quality_.level());
    while (effectCursor_ < battle.effects().size()) {
        const auto &effect = battle.effects()[effectCursor_++];
        using Type = sim::Effect::Type;
        switch (effect.type) {
        case Type::Muzzle:
            spawn(effect, density * 3, {215, 211, 190, 210}, 3.5f, .75f, .7f);
            break;
        case Type::Impact:
            spawn(effect, density * 3, {112, 83, 57, 255}, 5.0f, .8f, .42f);
            break;
        case Type::Splash:
            spawn(effect, density * 4, Foam, 4.0f, 1.0f, .42f);
            break;
        case Type::Destroy:
            spawn(effect, density * 3, {91, 60, 40, 255}, 5.5f, 1.2f, .38f);
            break;
        case Type::Crew:
            spawn(effect, density, {145, 48, 39, 220}, 2.0f, .55f, .3f);
            break;
        case Type::Sever:
            spawn(effect, density * 4, {84, 58, 38, 255}, 6.0f, 1.5f, .45f);
            break;
        case Type::Detonate:
            spawn(effect, density * 10, {236, 142, 58, 255}, 9.0f, 1.6f, .75f);
            shake_ = 1.0f;
            break;
        case Type::Ammo:
            break;
        }
    }
}

void DesktopRenderer::updateParticles(float seconds, double windTo) {
    const float windX = static_cast<float>(std::sin(windTo)) * .7f;
    const float windZ = static_cast<float>(-std::cos(windTo)) * .7f;
    stats_.particles = 0;
    shake_ = std::max(0.0f, shake_ - seconds * 1.8f);
    for (auto &particle : particles_) {
        if (!particle.active)
            continue;
        particle.age += seconds;
        if (particle.age >= particle.life) {
            particle.active = false;
            continue;
        }
        particle.velocity.x += windX * seconds;
        particle.velocity.z += windZ * seconds;
        particle.velocity.y -= 4.2f * seconds;
        particle.position = Vector3Add(particle.position, Vector3Scale(particle.velocity, seconds));
        if (particle.position.y < .08f) {
            particle.position.y = .08f;
            particle.velocity.y *= -.18f;
        }
        ++stats_.particles;
    }
}

void DesktopRenderer::drawPartDetail(const sim::RuntimeShip &ship, const sim::RuntimeCell &cell,
                                     Vector3 point, float heading, float sink, float damage) const {
    const Vector3 forward{std::sin(heading), 0, -std::cos(heading)};
    const Vector3 side{std::cos(heading), 0, std::sin(heading)};
    const Color dark = ColorAlpha({22, 27, 28, 255}, .9f);
    const auto type = static_cast<std::size_t>(cell.id);
    const float blockHeight =
        std::array<float, 10>{.20f, .28f, .26f, .30f, .28f, .32f, .34f, .30f, .36f, .30f}[type];
    point.y = .64f + blockHeight * damage - sink;
    const auto line = [&](Vector3 a, Vector3 b, float thick, Color color) {
        DrawCylinderEx(a, b, thick, thick * .72f, 7, color);
    };
    switch (cell.id) {
    case sim::PartId::Timber:
        for (int n = -1; n <= 1; ++n) {
            const Vector3 offset = Vector3Scale(side, n * .48f);
            line(Vector3Add(Vector3Add(point, offset), Vector3Scale(forward, -.65f)),
                 Vector3Add(Vector3Add(point, offset), Vector3Scale(forward, .65f)), .045f,
                 ColorAlpha(dark, .72f * damage));
        }
        break;
    case sim::PartId::Heavy:
        line(Vector3Add(point, Vector3Scale(side, -.65f)), Vector3Add(point, Vector3Scale(side, .65f)), .055f,
             {224, 232, 226, 220});
        line(Vector3Add(point, Vector3Scale(forward, -.65f)), Vector3Add(point, Vector3Scale(forward, .65f)),
             .055f, {224, 232, 226, 220});
        break;
    case sim::PartId::Crew:
        DrawSphere(Vector3Add(point, Vector3Scale(side, -.42f)), .20f, dark);
        DrawSphere(point, .20f, dark);
        DrawSphere(Vector3Add(point, Vector3Scale(side, .42f)), .20f, dark);
        break;
    case sim::PartId::Mast: {
        Vector3 base = point;
        base.y = .62f - sink;
        Vector3 top = base;
        top.y += 5.4f * damage;
        line(base, top, .13f, {202, 191, 166, 255});
        const Vector3 yardA = Vector3Add(Vector3Add(base, Vector3Scale(side, -1.35f)), {0, 3.6f, 0});
        const Vector3 yardB = Vector3Add(Vector3Add(base, Vector3Scale(side, 1.35f)), {0, 3.6f, 0});
        line(yardA, yardB, .08f, {202, 191, 166, 255});
        break;
    }
    case sim::PartId::Magazine:
        line(Vector3Add(Vector3Add(point, Vector3Scale(side, -.62f)), Vector3Scale(forward, -.62f)),
             Vector3Add(Vector3Add(point, Vector3Scale(side, .62f)), Vector3Scale(forward, .62f)), .06f,
             {255, 226, 166, 245});
        line(Vector3Add(Vector3Add(point, Vector3Scale(side, -.62f)), Vector3Scale(forward, .62f)),
             Vector3Add(Vector3Add(point, Vector3Scale(side, .62f)), Vector3Scale(forward, -.62f)), .06f,
             {255, 226, 166, 245});
        break;
    case sim::PartId::Swivel:
        DrawSphere(point, .30f, dark);
        line(point, Vector3Add(point, Vector3Scale(forward, .85f)), .10f, dark);
        break;
    case sim::PartId::GunDeck:
    case sim::PartId::Carronade: {
        const int barrels = cell.id == sim::PartId::GunDeck ? 3 : 2;
        const float direction = cell.coord.dx < 0 ? -1.0f : 1.0f;
        for (int i = 0; i < barrels; ++i) {
            Vector3 base = Vector3Add(point, Vector3Scale(forward, (i - (barrels - 1) * .5f) * .45f));
            line(base, Vector3Add(base, Vector3Scale(side, direction * .92f)),
                 cell.id == sim::PartId::Carronade ? .14f : .10f, dark);
        }
        break;
    }
    case sim::PartId::LongGun:
        line(Vector3Add(point, Vector3Scale(forward, -.35f)), Vector3Add(point, Vector3Scale(forward, 1.15f)),
             .13f, dark);
        break;
    case sim::PartId::Helm:
        DrawCircle3D(point, .48f, forward, 90, {44, 35, 28, 255});
        for (int i = 0; i < 8; ++i) {
            const float a = i * PI / 4.0f;
            Vector3 edge = Vector3Add(
                point, Vector3Add(Vector3Scale(side, std::cos(a) * .58f), Vector3{0, std::sin(a) * .58f, 0}));
            line(point, edge, .035f, {44, 35, 28, 255});
        }
        break;
    }
}

void DesktopRenderer::drawShips(const sim::Battle &battle) {
    partInstanceCounts_.fill(0);
    stats_.ships = static_cast<int>(battle.state().size());
    stats_.parts = 0;
    for (const auto &ship : battle.state()) {
        const float sink =
            ship.out ? std::min(2.2f, static_cast<float>((battle.time() - ship.outAt) * .35)) : 0.0f;
        const float heading = static_cast<float>(ship.heading);
        const auto &hull = sim::hull(ship.hullIndex);
        const auto seat = static_cast<std::size_t>(ship.index % 4);
        drawEllipse({static_cast<float>(ship.position.x), .05f, static_cast<float>(ship.position.z)},
                    (hull.width / 2.0f + .62f) * static_cast<float>(sim::CellSize),
                    (hull.length / 2.0f + .78f) * static_cast<float>(sim::CellSize), heading,
                    ColorAlpha(PlayerColors[seat], .20f));

        for (const auto &coord : hull.cells) {
            const auto found = std::find_if(ship.cells.begin(), ship.cells.end(),
                                            [&](const auto &cell) { return cell.coord == coord; });
            const bool occupied = found != ship.cells.end();
            const bool alive = occupied && found->alive;
            // A permanent under-deck keeps the tapered hull readable after fitted cells are
            // destroyed. Per-cell plates preserve the construction grid instead of merging each
            // row into an ambiguous rectangular bar.
            const Vector3 body = worldCell(ship, coord, .18f - sink);
            DrawMesh(cube_, hullMaterials_[seat],
                     transform(body.x, body.y, body.z, 2.28f, .36f, 2.28f, -heading));
            const Vector3 deck = worldCell(ship, coord, alive ? .42f - sink : .23f - sink);
            const Color base = coord.dx == 0 ? SpineColors[seat] : DeckColors[seat];
            const Color color = alive ? base : Color{16, 24, 32, 255};
            DrawCubeV(deck, {2.12f, alive ? .28f : .12f, 2.12f}, color);
            DrawCubeWiresV(deck, {2.12f, alive ? .28f : .12f, 2.12f}, ColorAlpha({8, 13, 17, 255}, .82f));
        }

        const float bowFore = (hull.bowZ - .78f) * static_cast<float>(sim::CellSize);
        const Vector3 bowBase{
            static_cast<float>(ship.position.x) - static_cast<float>(ship.sinHeading) * bowFore, .53f - sink,
            static_cast<float>(ship.position.z) + static_cast<float>(ship.cosHeading) * bowFore};
        const Vector3 side{static_cast<float>(ship.cosHeading), 0, static_cast<float>(ship.sinHeading)};
        const Vector3 forward{static_cast<float>(ship.sinHeading), 0, -static_cast<float>(ship.cosHeading)};
        DrawTriangle3D(Vector3Add(bowBase, Vector3Scale(side, -1.15f)),
                       Vector3Add(bowBase, Vector3Scale(forward, 2.45f)),
                       Vector3Add(bowBase, Vector3Scale(side, 1.15f)), HullColors[seat]);
        const Vector3 bowspritBase = Vector3Add(bowBase, Vector3{0, .15f, 0});
        DrawCylinderEx(bowspritBase, Vector3Add(bowspritBase, Vector3Scale(forward, 3.15f)), .075f, .035f, 7,
                       {224, 211, 178, 255});

        for (const auto &cell : ship.cells) {
            if (!cell.alive)
                continue;
            const Vector3 point = worldCell(ship, cell.coord, .58f - sink);
            const float damage = static_cast<float>(std::clamp(cell.hp / cell.maxHp, .25, 1.0));
            const auto type = static_cast<std::size_t>(cell.id);
            auto &count = partInstanceCounts_[type];
            const float height =
                std::array<float, 10>{.20f, .28f, .26f, .30f, .28f, .32f, .34f, .30f, .36f, .30f}[type];
            if (count < static_cast<int>(partInstances_[type].size()))
                partInstances_[type][static_cast<std::size_t>(count++)] =
                    transform(point.x, point.y + height * damage * .5f, point.z, 1.72f, height * damage,
                              1.72f, -heading);
            ++stats_.parts;
            drawPartDetail(ship, cell, point, heading, sink, damage);
        }

        const float sternFore = (hull.cells.back().dz + .68f) * static_cast<float>(sim::CellSize);
        Vector3 pole{static_cast<float>(ship.position.x) - static_cast<float>(ship.sinHeading) * sternFore,
                     .55f - sink,
                     static_cast<float>(ship.position.z) + static_cast<float>(ship.cosHeading) * sternFore};
        Vector3 poleTop = pole;
        poleTop.y += 4.2f;
        DrawCylinderEx(pole, poleTop, .09f, .09f, 7, {202, 191, 166, 255});
        DrawTriangle3D(poleTop, Vector3Add(Vector3Add(poleTop, Vector3Scale(side, 2.0f)), {0, -.55f, 0}),
                       Vector3Add(poleTop, {0, -1.1f, 0}), PlayerColors[seat]);
    }
    for (std::size_t type = 0; type < partInstances_.size(); ++type)
        if (partInstanceCounts_[type] > 0)
            DrawMeshInstanced(cube_, partMaterials_[type], partInstances_[type].data(),
                              partInstanceCounts_[type]);
}

void DesktopRenderer::drawParticles(const Camera3D &camera) const {
    (void)camera;
    for (const auto &particle : particles_) {
        if (!particle.active)
            continue;
        const float remaining = 1.0f - particle.age / particle.life;
        if (particle.mast) {
            const Vector3 end{particle.position.x + std::sin(particle.age * 2.2f) * 2.8f,
                              particle.position.y + particle.size * remaining,
                              particle.position.z + std::cos(particle.age * 2.2f) * 2.8f};
            DrawCylinderEx(particle.position, end, .12f, .08f, 7, ColorAlpha({91, 63, 40, 255}, remaining));
        } else {
            DrawSphere(particle.position, particle.size * remaining * .42f,
                       ColorAlpha(particle.color, remaining));
        }
    }
}

void DesktopRenderer::drawBattle(const sim::Battle &battle, int screenWidth, int screenHeight,
                                 float frameSeconds) {
    quality_.sample(frameSeconds);
    ensureTarget(screenWidth, screenHeight);
    stats_.renderScale = targetWidth_ / static_cast<float>(std::max(1, screenWidth));
    consumeEffects(battle);
    updateParticles(std::min(frameSeconds, .05f), battle.windTo());

    Vector3 desired{};
    int active = 0;
    float extent = 30.0f;
    std::array<const sim::RuntimeShip *, sim::MaxPlayers> framed{};
    std::size_t framedCount = 0;
    for (const auto &ship : battle.state()) {
        if (ship.out && battle.time() - ship.outAt > 2.2)
            continue;
        framed[framedCount++] = &ship;
        if (ship.out)
            continue;
        desired.x += static_cast<float>(ship.position.x);
        desired.z += static_cast<float>(ship.position.z);
        ++active;
    }
    if (active) {
        desired.x /= active;
        desired.z /= active;
    }
    float spread = 0.0f;
    for (std::size_t i = 0; i < framedCount; ++i)
        for (std::size_t j = i + 1; j < framedCount; ++j)
            spread = std::max(spread, Vector2Distance({static_cast<float>(framed[i]->position.x),
                                                       static_cast<float>(framed[i]->position.z)},
                                                      {static_cast<float>(framed[j]->position.x),
                                                       static_cast<float>(framed[j]->position.z)}));
    extent = std::clamp(spread * .62f + 13.0f, 24.0f, 78.0f) * 2.0f;
    if (cameraSeed_ != battle.seed()) {
        cameraSeed_ = battle.seed();
        cameraTarget_ = desired;
        cameraSpan_ = extent;
    } else {
        cameraTarget_ = Vector3Lerp(cameraTarget_, desired, 1.0f - std::pow(.001f, frameSeconds));
        cameraSpan_ += (extent - cameraSpan_) * (1.0f - std::pow(.02f, frameSeconds));
    }
    Camera3D camera{};
    camera.position = {cameraTarget_.x, cameraTarget_.y + 260.0f * std::sin(60.0f * DEG2RAD),
                       cameraTarget_.z + 260.0f * std::cos(60.0f * DEG2RAD)};
    camera.target = cameraTarget_;
    if (shake_ > 0) {
        const float phase = static_cast<float>(battle.time()) * 71.0f;
        camera.position.x += std::sin(phase) * shake_ * 1.4f;
        camera.position.z += std::cos(phase * 1.31f) * shake_ * 1.4f;
        camera.target.x += std::cos(phase * .83f) * shake_ * .7f;
    }
    camera.up = {0, 1, 0};
    camera.fovy = cameraSpan_;
    camera.projection = CAMERA_ORTHOGRAPHIC;

    BeginTextureMode(target_);
    ClearBackground(SeaDeep);
    drawSeaIntoTarget(static_cast<float>(battle.time()), battle.windTo(), cameraTarget_, cameraSpan_, 60.0f);
    BeginMode3D(camera);
    drawShips(battle);
    stats_.projectiles = std::min(400, static_cast<int>(battle.projectiles().size()));
    for (int i = 0; i < stats_.projectiles; ++i) {
        const auto &shot = battle.projectiles()[static_cast<std::size_t>(i)];
        DrawSphere({static_cast<float>(shot.pos.x), .85f, static_cast<float>(shot.pos.z)},
                   shot.kind == sim::Ammo::Round ? .18f : .11f,
                   shot.kind == sim::Ammo::Round ? Color{35, 31, 28, 255} : Color{89, 64, 45, 255});
        const Vector3 head{static_cast<float>(shot.pos.x), .85f, static_cast<float>(shot.pos.z)};
        const Vector3 tail{head.x - static_cast<float>(shot.velocity.x) * .045f, .82f,
                           head.z - static_cast<float>(shot.velocity.z) * .045f};
        DrawLine3D(head, tail,
                   shot.kind == sim::Ammo::Round ? Color{255, 220, 166, 120} : Color{200, 180, 138, 100});
    }
    drawParticles(camera);
    EndMode3D();
    EndTextureMode();

    presentTarget(screenWidth, screenHeight);
}

} // namespace broadside::render
