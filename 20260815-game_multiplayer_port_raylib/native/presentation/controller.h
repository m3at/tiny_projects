#pragma once
#include "net/session.h"
#include <memory>
namespace broadside::presentation {
struct Rect {
    float x = 0, y = 0, w = 0, h = 0;
    bool contains(float px, float py) const {
        return px >= x && py >= y && px < x + w && py < y + h;
    }
};
struct InputState {
    float mouseX = 0, mouseY = 0;
    bool primary = false, secondary = false, confirm = false, back = false;
};
struct UiLayout {
    int width = 1280, height = 720;
    float scale = 1;
    Rect lock{}, reroll{}, refit{}, roundAmmo{}, grapeAmmo{};
    static UiLayout make(int width, int height);
};
struct UiModel {
    int phase = 0, round = 0, activeSeat = -1;
    float deadline = 0;
};
class AppController {
  public:
    void start(std::uint32_t seed, int players, int bots) {
        session_ = std::make_unique<net::LocalSession>(seed, players, bots);
    }
    void update(float seconds) {
        if (session_)
            session_->update(seconds);
    }
    bool started() const {
        return static_cast<bool>(session_);
    }
    net::LocalSession &session() {
        return *session_;
    }
    const net::LocalSession &session() const {
        return *session_;
    }
    UiModel model() const;

  private:
    std::unique_ptr<net::LocalSession> session_;
};
} // namespace broadside::presentation
