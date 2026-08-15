#include "audio.h"
#include "sim/rng.h"
#include <algorithm>
#include <cmath>
#include <numbers>

namespace broadside::audio {
namespace {
constexpr float Tau = 6.283185307179586f;
}
std::vector<float> synthesize(Cue cue, float sampleRate, std::uint32_t seed) {
    const bool ambience = cue == Cue::Wind || cue == Cue::Sea;
    const bool large = cue == Cue::Detonation || cue == Cue::MastBreak;
    const float seconds = ambience ? 1.0f : (large ? .5f : .18f);
    std::vector<float> out(static_cast<std::size_t>(seconds * sampleRate));
    sim::Rng rng(seed ^ static_cast<std::uint32_t>(cue));
    float frequency = 90, decay = 18, gain = .28f;
    switch (cue) {
    case Cue::Cannon:
        frequency = 72;
        decay = 18;
        gain = .35f;
        break;
    case Cue::RoundImpact:
        frequency = 120;
        decay = 24;
        break;
    case Cue::GrapeImpact:
        frequency = 190;
        decay = 30;
        break;
    case Cue::Splinter:
        frequency = 310;
        decay = 35;
        break;
    case Cue::Splash:
        frequency = 150;
        decay = 12;
        break;
    case Cue::MastBreak:
        frequency = 55;
        decay = 8;
        break;
    case Cue::Detonation:
        frequency = 42;
        decay = 7;
        gain = .42f;
        break;
    case Cue::Select:
        frequency = 520;
        decay = 28;
        gain = .16f;
        break;
    case Cue::Place:
        frequency = 360;
        decay = 24;
        gain = .18f;
        break;
    case Cue::Press:
        frequency = 440;
        decay = 30;
        gain = .15f;
        break;
    case Cue::Confirm:
        frequency = 660;
        decay = 18;
        gain = .16f;
        break;
    case Cue::Deny:
        frequency = 145;
        decay = 16;
        gain = .16f;
        break;
    case Cue::Wind:
        frequency = 36;
        decay = 0;
        gain = .08f;
        break;
    case Cue::Sea:
        frequency = 28;
        decay = 0;
        gain = .07f;
        break;
    }
    float noiseState = 0;
    for (std::size_t i = 0; i < out.size(); ++i) {
        float t = static_cast<float>(i) / sampleRate;
        float attack = std::min(1.0f, t / .004f);
        float envelope =
            ambience ? std::sin(std::min(1.0f, t / .08f) * 1.5707963f) : attack * std::exp(-decay * t);
        float noise = rng.range(-1, 1);
        noiseState = noiseState * .9f + noise * .1f;
        float tone = std::sin(Tau * (frequency - frequency * .12f * t) * t);
        out[i] = gain * envelope * (tone * .72f + noiseState * .28f);
    }
    return out;
}
std::vector<float> cannon(float seconds, float sampleRate) {
    auto out = synthesize(Cue::Cannon, sampleRate, 1);
    out.resize(std::min(out.size(), static_cast<std::size_t>(seconds * sampleRate)));
    return out;
}
std::vector<float> mixedBurst(float sampleRate) {
    return worstCaseBurst(sampleRate);
}

Mixer::Mixer(float sampleRate) : sampleRate_(sampleRate) {
    for (int cue = 0; cue < CueCount; ++cue)
        bank_[static_cast<std::size_t>(cue)] =
            synthesize(static_cast<Cue>(cue), sampleRate_, static_cast<std::uint32_t>(cue + 1));
}

bool Mixer::enqueue(SpatialEvent event) {
    ++stats_.submitted;
    event.pan = std::clamp(event.pan, -1.0f, 1.0f);
    event.gain = std::clamp(event.gain, 0.0f, 2.0f);
    // Cannon events are rolled across 90 ms. Dense broadsides remain articulate
    // instead of collapsing into a single clipped transient.
    if (event.cue == Cue::Cannon) {
        event.delay += static_cast<float>(volleyIndex_ % 8) * .012f;
        ++volleyIndex_;
    }
    for (auto &pending : queue_)
        if (!pending.active) {
            pending.event = event;
            pending.delayFrames = static_cast<std::uint32_t>(std::max(0.0f, event.delay) * sampleRate_);
            pending.active = true;
            return true;
        }
    ++stats_.queueDrops;
    return false;
}

void Mixer::start(const SpatialEvent &event) {
    Voice *slot = nullptr;
    for (auto &voice : voices_)
        if (!voice.active) {
            slot = &voice;
            break;
        }
    if (!slot) {
        slot = &*std::max_element(voices_.begin(), voices_.end(),
                                  [](const Voice &a, const Voice &b) { return a.cursor < b.cursor; });
        ++stats_.voiceSteals;
    }
    const float angle = (event.pan + 1.0f) * static_cast<float>(std::numbers::pi) / 4.0f;
    slot->cue = event.cue;
    slot->cursor = 0;
    slot->left = std::cos(angle) * event.gain;
    slot->right = std::sin(angle) * event.gain;
    slot->active = true;
    ++stats_.started;
}

void Mixer::renderInto(float *output, std::size_t frames) {
    std::fill(output, output + frames * 2, 0.0f);
    for (std::size_t frame = 0; frame < frames; ++frame) {
        for (auto &pending : queue_)
            if (pending.active) {
                if (pending.delayFrames > 0)
                    --pending.delayFrames;
                else {
                    start(pending.event);
                    pending.active = false;
                }
            }
        int transients = 0;
        for (const auto &voice : voices_)
            if (voice.active && voice.cue != Cue::Wind && voice.cue != Cue::Sea)
                ++transients;
        const float ambienceDuck = 1.0f / (1.0f + transients * .13f);
        float left = 0, right = 0;
        for (auto &voice : voices_) {
            if (!voice.active)
                continue;
            const auto &samples = bank_[static_cast<std::size_t>(voice.cue)];
            if (voice.cursor >= samples.size()) {
                voice.active = false;
                continue;
            }
            float sample = samples[voice.cursor++];
            if (voice.cue == Cue::Wind || voice.cue == Cue::Sea)
                sample *= ambienceDuck;
            left += sample * voice.left;
            right += sample * voice.right;
        }
        // Soft limiting has unity slope near silence and cannot clip.
        left = std::tanh(left * .82f);
        right = std::tanh(right * .82f);
        stats_.peak = std::max(stats_.peak, std::max(std::abs(left), std::abs(right)));
        output[frame * 2] = left;
        output[frame * 2 + 1] = right;
    }
}

std::size_t Mixer::activeVoices() const {
    return static_cast<std::size_t>(
        std::count_if(voices_.begin(), voices_.end(), [](const Voice &voice) { return voice.active; }));
}

std::vector<float> broadsideBurst(float sampleRate) {
    Mixer mixer(sampleRate);
    for (int i = 0; i < 16; ++i)
        mixer.enqueue({Cue::Cannon, -1.0f + 2.0f * (i / 15.0f), .54f, 0});
    std::vector<float> out(static_cast<std::size_t>(1.0f * sampleRate) * 2);
    mixer.renderInto(out.data(), out.size() / 2);
    return out;
}

std::vector<float> worstCaseBurst(float sampleRate, MixerStats *stats) {
    Mixer mixer(sampleRate);
    constexpr Cue dense[]{Cue::Cannon, Cue::RoundImpact, Cue::GrapeImpact, Cue::Splinter,
                          Cue::Splash, Cue::MastBreak,   Cue::Detonation};
    mixer.enqueue({Cue::Wind, 0, .35f, 0});
    mixer.enqueue({Cue::Sea, 0, .35f, 0});
    for (int i = 0; i < 64; ++i)
        mixer.enqueue({dense[i % 7], -1.0f + 2.0f * ((i % 17) / 16.0f), .34f, (i % 9) * .004f});
    std::vector<float> out(static_cast<std::size_t>(1.2f * sampleRate) * 2);
    mixer.renderInto(out.data(), out.size() / 2);
    if (stats)
        *stats = mixer.stats();
    return out;
}
} // namespace broadside::audio
