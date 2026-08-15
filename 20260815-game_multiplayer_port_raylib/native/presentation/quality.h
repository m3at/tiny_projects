#pragma once

#include <array>

namespace broadside::presentation {

// Presentation quality is deliberately driven by a smoothed frame cost, never by
// simulation time. This keeps the 60 Hz battle deterministic while allowing the
// render target and effect density to adapt to the local GPU.
class AdaptiveQuality {
  public:
    static constexpr std::array<float, 5> Scales{1.0f, 0.85f, 0.72f, 0.60f, 0.50f};

    void sample(float frameSeconds);
    [[nodiscard]] float scale() const {
        return Scales[index_];
    }
    [[nodiscard]] int level() const {
        return index_;
    }
    [[nodiscard]] float smoothedFrameSeconds() const {
        return smoothed_;
    }
    void reset();

  private:
    int index_ = 0;
    float smoothed_ = 1.0f / 60.0f;
    float slowFor_ = 0.0f;
    float fastFor_ = 0.0f;
};

} // namespace broadside::presentation
