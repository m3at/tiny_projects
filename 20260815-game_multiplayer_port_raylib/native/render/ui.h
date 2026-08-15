#pragma once

#include "raylib.h"

#include <string_view>

namespace broadside::render {

namespace theme {
inline constexpr Color Ink{233, 228, 216, 255};
inline constexpr Color Dim{181, 172, 155, 255};
inline constexpr Color Gold{216, 178, 92, 255};
inline constexpr Color Panel{12, 18, 25, 226};
inline constexpr Color Gutter{5, 9, 14, 210};
inline constexpr Color Rule{233, 228, 216, 51};
inline constexpr Color RuleSoft{233, 228, 216, 21};
inline constexpr Color Sound{126, 201, 138, 255};
inline constexpr Color Mauled{224, 193, 100, 255};
inline constexpr Color Sinking{224, 122, 100, 255};
inline constexpr Color Players[4]{
    {95, 168, 255, 255}, {255, 122, 95, 255}, {99, 209, 168, 255}, {201, 140, 240, 255}};
inline constexpr Color Parts[10]{{138, 104, 68, 255},  {77, 90, 94, 255},   {201, 162, 39, 255},
                                 {216, 203, 176, 255}, {176, 48, 48, 255},  {127, 168, 201, 255},
                                 {74, 111, 165, 255},  {139, 95, 176, 255}, {47, 143, 111, 255},
                                 {232, 232, 232, 255}};
} // namespace theme

class UiPainter {
  public:
    UiPainter(Font heading, Font body, int width, int height);

    [[nodiscard]] float scale() const {
        return scale_;
    }
    [[nodiscard]] int px(float value) const;
    [[nodiscard]] Rectangle rect(float x, float y, float width, float height) const;
    [[nodiscard]] float measure(std::string_view text, float size, bool display = false) const;
    void text(std::string_view value, float x, float y, float size, Color color = theme::Ink,
              bool display = false) const;
    void textRight(std::string_view value, float right, float y, float size, Color color = theme::Ink,
                   bool display = false) const;
    void textFit(std::string_view value, float x, float y, float maxWidth, float size,
                 Color color = theme::Ink, bool display = false) const;
    void textWrapped(std::string_view value, float x, float y, float maxWidth, float size, float lineHeight,
                     int maxLines, Color color = theme::Ink, bool display = false) const;
    void panel(Rectangle bounds, Color accent = BLANK, bool accentRight = false) const;
    void section(std::string_view label, float x, float y, float width) const;
    bool button(Rectangle bounds, std::string_view label, bool enabled = true, bool selected = false,
                Color accent = theme::Gold) const;
    void progress(Rectangle bounds, float fraction) const;
    void compass(Vector2 centre, float radius, double bearing) const;
    void cornerTitle(std::string_view title, std::string_view subtitle) const;

  private:
    Font heading_{};
    Font body_{};
    int width_ = 0;
    int height_ = 0;
    float scale_ = 1;
};

} // namespace broadside::render
