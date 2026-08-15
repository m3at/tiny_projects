#include "quality.h"

#include <algorithm>

namespace broadside::presentation {

void AdaptiveQuality::sample(float frameSeconds) {
    frameSeconds = std::clamp(frameSeconds, 0.001f, 0.25f);
    smoothed_ += (frameSeconds - smoothed_) * 0.06f;
    if (smoothed_ > 1.0f / 49.0f) {
        slowFor_ += frameSeconds;
        fastFor_ = 0.0f;
    } else if (smoothed_ < 1.0f / 58.0f) {
        fastFor_ += frameSeconds;
        slowFor_ = 0.0f;
    } else {
        slowFor_ = std::max(0.0f, slowFor_ - frameSeconds * 0.5f);
        fastFor_ = std::max(0.0f, fastFor_ - frameSeconds * 0.5f);
    }
    if (slowFor_ >= 0.75f && index_ + 1 < static_cast<int>(Scales.size())) {
        ++index_;
        slowFor_ = fastFor_ = 0.0f;
    } else if (fastFor_ >= 3.0f && index_ > 0) {
        --index_;
        slowFor_ = fastFor_ = 0.0f;
    }
}

void AdaptiveQuality::reset() {
    index_ = 0;
    smoothed_ = 1.0f / 60.0f;
    slowFor_ = fastFor_ = 0.0f;
}

} // namespace broadside::presentation
