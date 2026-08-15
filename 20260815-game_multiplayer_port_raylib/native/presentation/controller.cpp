#include "controller.h"
#include <algorithm>
namespace broadside::presentation {
UiLayout UiLayout::make(int width, int height) {
    UiLayout layout;
    layout.width = width;
    layout.height = height;
    layout.scale = std::min(width / 1280.0f, height / 720.0f);
    const float s = layout.scale;
    layout.lock = {32 * s, height - 72 * s, 170 * s, 40 * s};
    layout.reroll = {214 * s, height - 72 * s, 120 * s, 40 * s};
    layout.refit = {346 * s, height - 72 * s, 120 * s, 40 * s};
    layout.roundAmmo = {width - 250 * s, height - 72 * s, 100 * s, 40 * s};
    layout.grapeAmmo = {width - 140 * s, height - 72 * s, 100 * s, 40 * s};
    return layout;
}
UiModel AppController::model() const {
    UiModel model;
    if (!session_)
        return model;
    const auto &state = session_->state();
    model.phase = state.phase;
    model.round = state.round;
    model.activeSeat = state.activeSeat;
    if (state.activeSeat >= 0 && state.activeSeat < session_->humans()) {
        const auto &build = session_->client(state.activeSeat).state().build;
        if (build)
            model.deadline = build->deadline;
    }
    return model;
}
} // namespace broadside::presentation
