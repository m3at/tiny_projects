#pragma once

#include "presentation/battle_visual.h"
#include "presentation/quality.h"
#include "raylib.h"
#include "sim/battle.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace broadside::render {

struct RenderStats {
    int ships = 0;
    int parts = 0;
    int projectiles = 0;
    int particles = 0;
    int particleOverflow = 0;
    int flags = 0;
    int sinkingShips = 0;
    float renderScale = 1.0f;
};

class DesktopRenderer {
  public:
    DesktopRenderer();
    ~DesktopRenderer();
    DesktopRenderer(const DesktopRenderer &) = delete;
    DesktopRenderer &operator=(const DesktopRenderer &) = delete;
    void unload();

    void drawBattle(const sim::Battle &battle, int screenWidth, int screenHeight, float frameSeconds);
    void drawSea(int screenWidth, int screenHeight, float time, double windTo);
    void resetEffects();

    [[nodiscard]] Font headingFont() const {
        return heading_;
    }
    [[nodiscard]] Font bodyFont() const {
        return body_;
    }
    [[nodiscard]] const RenderStats &stats() const {
        return stats_;
    }
    [[nodiscard]] const presentation::AdaptiveQuality &quality() const {
        return quality_;
    }

  private:
    struct Particle {
        Vector3 position{};
        Vector3 velocity{};
        float age = 0.0f;
        float life = 0.0f;
        float size = 0.0f;
        Color color{};
        bool active = false;
        bool mast = false;
    };

    void ensureTarget(int width, int height);
    void consumeEffects(const sim::Battle &battle);
    void spawn(const sim::Effect &effect, int count, Color color, float speed, float life, float size);
    void updateParticles(float seconds, double windTo);
    void updateBattlePresentation(const sim::Battle &battle, float seconds);
    void drawShips(const sim::Battle &battle);
    void drawParticles(const Camera3D &camera) const;
    void drawSeaIntoTarget(float time, double windTo, Vector3 centre, float viewHeight, float arenaRadius);
    void presentTarget(int screenWidth, int screenHeight);
    void drawPartDetail(const sim::RuntimeShip &ship, const sim::RuntimeCell &cell, Vector3 point,
                        float heading, float sink, float damage) const;

    RenderTexture2D target_{};
    Shader seaShader_{};
    Shader postShader_{};
    Mesh cube_{};
    std::array<Material, 10> partMaterials_{};
    std::array<Material, 4> hullMaterials_{};
    std::array<Material, 4> deckMaterials_{};
    std::array<Material, 4> spineMaterials_{};
    Material holeMaterial_{};
    Material wreckMaterial_{};
    Font heading_{};
    Font body_{};
    std::array<Particle, 720> particles_{};
    std::array<std::array<Matrix, 256>, 10> partInstances_{};
    std::array<int, 10> partInstanceCounts_{};
    std::size_t nextParticle_ = 0;
    std::size_t effectCursor_ = 0;
    std::uint32_t effectSeed_ = 0;
    bool cameraInitialized_ = false;
    Vector3 cameraTarget_{};
    float cameraSpan_ = 80.0f;
    float shake_ = 0.0f;
    int targetWidth_ = 0;
    int targetHeight_ = 0;
    int resolutionLocation_ = -1;
    int timeLocation_ = -1;
    int windLocation_ = -1;
    int mapLocation_ = -1;
    int pixelLocation_ = -1;
    int ringLocation_ = -1;
    int inverseResolutionLocation_ = -1;
    presentation::BattleVisualState battleVisual_;
    presentation::AdaptiveQuality quality_;
    RenderStats stats_{};
    bool loaded_ = true;
};

} // namespace broadside::render
