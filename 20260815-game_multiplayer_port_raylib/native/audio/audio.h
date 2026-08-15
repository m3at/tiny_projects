#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
namespace broadside::audio {
enum class Cue {
    Cannon,
    RoundImpact,
    GrapeImpact,
    Splinter,
    Splash,
    MastBreak,
    Detonation,
    Select,
    Place,
    Press,
    Confirm,
    Deny,
    Wind,
    Sea
};
constexpr int CueCount = static_cast<int>(Cue::Sea) + 1;
std::vector<float> synthesize(Cue cue, float sampleRate = 48000.0f, std::uint32_t seed = 1);
std::vector<float> cannon(float seconds = 0.18f, float sampleRate = 48000.0f);
std::vector<float> mixedBurst(float sampleRate = 48000.0f);

struct SpatialEvent {
    Cue cue = Cue::Press;
    float pan = 0.0f; // -1 left, +1 right
    float gain = 1.0f;
    float delay = 0.0f;
};

struct MixerStats {
    std::uint32_t submitted = 0;
    std::uint32_t started = 0;
    std::uint32_t queueDrops = 0;
    std::uint32_t voiceSteals = 0;
    float peak = 0.0f;
};

// A fixed-capacity headless mixer mirrors the desktop voice policy. Waveforms are
// prepared at construction; enqueue and renderInto allocate nothing.
class Mixer {
  public:
    explicit Mixer(float sampleRate = 48000.0f);
    bool enqueue(SpatialEvent event);
    void renderInto(float *stereoInterleaved, std::size_t frames);
    [[nodiscard]] const MixerStats &stats() const {
        return stats_;
    }
    [[nodiscard]] std::size_t activeVoices() const;

  private:
    struct Pending {
        SpatialEvent event{};
        std::uint32_t delayFrames = 0;
        bool active = false;
    };
    struct Voice {
        Cue cue = Cue::Press;
        std::size_t cursor = 0;
        float left = 0;
        float right = 0;
        bool active = false;
    };
    void start(const SpatialEvent &event);

    float sampleRate_ = 48000.0f;
    std::array<std::vector<float>, CueCount> bank_{};
    std::array<Pending, 128> queue_{};
    std::array<Voice, 48> voices_{};
    std::size_t volleyIndex_ = 0;
    MixerStats stats_{};
};

std::vector<float> broadsideBurst(float sampleRate = 48000.0f);
std::vector<float> worstCaseBurst(float sampleRate = 48000.0f, MixerStats *stats = nullptr);
} // namespace broadside::audio
