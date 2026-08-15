#include "ui.h"

#include <algorithm>
#include <cmath>
#include <string>

namespace broadside::render {

namespace {

bool useDisplayFace(float size, bool display) {
    // The display face is deliberately rough and loses counters at HUD sizes. Inter is the
    // legibility face; reserve IM FELL for text large enough to retain its character.
    return display && size >= 17.0f;
}

float textPixels(float size, float scale) {
    const float optical = size < 40 ? 1.2f : 1.0f;
    return std::max(12.5f, size * optical * scale);
}

} // namespace

UiPainter::UiPainter(Font heading, Font body, int width, int height)
    : heading_(heading), body_(body), width_(width), height_(height),
      scale_(std::clamp(std::min(width / 1280.0f, height / 720.0f), .75f, 1.45f)) {}

int UiPainter::px(float value) const {
    return static_cast<int>(std::round(value * scale_));
}

Rectangle UiPainter::rect(float x, float y, float width, float height) const {
    return {x * scale_, y * scale_, width * scale_, height * scale_};
}

float UiPainter::measure(std::string_view textValue, float size, bool display) const {
    const std::string text{textValue};
    const Font font = useDisplayFace(size, display) ? heading_ : body_;
    return MeasureTextEx(font, text.c_str(), textPixels(size, scale_), .5f * scale_).x;
}

void UiPainter::text(std::string_view value, float x, float y, float size, Color color, bool display) const {
    const std::string copy{value};
    const Font font = useDisplayFace(size, display) ? heading_ : body_;
    DrawTextEx(font, copy.c_str(), {x * scale_, y * scale_}, textPixels(size, scale_), .5f * scale_, color);
}

void UiPainter::textRight(std::string_view value, float right, float y, float size, Color color,
                          bool display) const {
    text(value, right - measure(value, size, display) / scale_, y, size, color, display);
}

void UiPainter::textFit(std::string_view value, float x, float y, float maxWidth, float size, Color color,
                        bool display) const {
    if (measure(value, size, display) <= maxWidth * scale_) {
        text(value, x, y, size, color, display);
        return;
    }
    std::string fitted{value};
    while (!fitted.empty() && measure(fitted + "...", size, display) > maxWidth * scale_)
        fitted.pop_back();
    fitted += "...";
    text(fitted, x, y, size, color, display);
}

void UiPainter::textWrapped(std::string_view value, float x, float y, float maxWidth, float size,
                            float lineHeight, int maxLines, Color color, bool display) const {
    std::string remaining{value};
    for (int line = 0; line < maxLines && !remaining.empty(); ++line) {
        std::size_t take = remaining.size();
        while (take > 0 &&
               measure(std::string_view(remaining).substr(0, take), size, display) > maxWidth * scale_) {
            const auto space = remaining.rfind(' ', take - 1);
            take = space == std::string::npos ? take - 1 : space;
        }
        if (take == 0)
            break;
        std::string row = remaining.substr(0, take);
        remaining.erase(0, take);
        while (!remaining.empty() && remaining.front() == ' ')
            remaining.erase(remaining.begin());
        if (line == maxLines - 1 && !remaining.empty()) {
            while (!row.empty() && measure(row + "...", size, display) > maxWidth * scale_)
                row.pop_back();
            row += "...";
        }
        text(row, x, y + line * lineHeight, size, color, display);
    }
}

void UiPainter::panel(Rectangle bounds, Color accent, bool accentRight) const {
    DrawRectangleRec(bounds, theme::Panel);
    DrawRectangleLinesEx(bounds, std::max(1.0f, scale_), theme::Rule);
    Rectangle inner{bounds.x + 4 * scale_, bounds.y + 4 * scale_, bounds.width - 8 * scale_,
                    bounds.height - 8 * scale_};
    DrawRectangleLinesEx(inner, std::max(1.0f, scale_), theme::RuleSoft);
    if (accent.a) {
        const float x = accentRight ? bounds.x + bounds.width - 3 * scale_ : bounds.x;
        DrawRectangleRec({x, bounds.y, 3 * scale_, bounds.height}, accent);
    }
    const float tick = 8 * scale_;
    DrawLineEx({bounds.x, bounds.y}, {bounds.x + tick, bounds.y}, 1, theme::Gold);
    DrawLineEx({bounds.x, bounds.y}, {bounds.x, bounds.y + tick}, 1, theme::Gold);
    DrawLineEx({bounds.x + bounds.width, bounds.y + bounds.height},
               {bounds.x + bounds.width - tick, bounds.y + bounds.height}, 1, theme::Gold);
    DrawLineEx({bounds.x + bounds.width, bounds.y + bounds.height},
               {bounds.x + bounds.width, bounds.y + bounds.height - tick}, 1, theme::Gold);
}

void UiPainter::section(std::string_view label, float x, float y, float width) const {
    text(label, x, y, 11, theme::Dim, true);
    const float start = x * scale_ + measure(label, 11, true) + 10 * scale_;
    DrawLineEx({start, (y + 7) * scale_}, {(x + width) * scale_, (y + 7) * scale_}, 1, theme::Rule);
}

bool UiPainter::button(Rectangle bounds, std::string_view label, bool enabled, bool selected,
                       Color accent) const {
    const bool hover = enabled && CheckCollisionPointRec(GetMousePosition(), bounds);
    const Color fill = !enabled   ? Color{18, 24, 29, 225}
                       : selected ? ColorAlpha(accent, .80f)
                       : hover    ? Color{38, 51, 57, 245}
                                  : Color{19, 29, 36, 235};
    DrawRectangleRec(bounds, fill);
    DrawRectangleLinesEx(bounds, std::max(1.0f, scale_),
                         enabled ? (selected || hover ? accent : theme::Rule) : theme::RuleSoft);
    DrawRectangleRec({bounds.x, bounds.y, 2 * scale_, bounds.height}, enabled ? accent : theme::RuleSoft);
    const Color ink = !enabled   ? ColorAlpha(theme::Dim, .55f)
                      : selected ? Color{18, 18, 14, 255}
                                 : theme::Ink;
    const float size = std::max(13.0f, 15 * scale_);
    const std::string copy{label};
    const Vector2 measured = MeasureTextEx(body_, copy.c_str(), size, .5f * scale_);
    DrawTextEx(body_, copy.c_str(),
               {bounds.x + (bounds.width - measured.x) * .5f,
                bounds.y + (bounds.height - measured.y) * .5f - scale_},
               size, .5f * scale_, ink);
    return hover && IsMouseButtonPressed(MOUSE_BUTTON_LEFT);
}

void UiPainter::progress(Rectangle bounds, float fraction) const {
    fraction = std::clamp(fraction, 0.0f, 1.0f);
    DrawRectangleRec(bounds, Color{2, 5, 8, 190});
    const Color fill = fraction > .58f ? theme::Sound : fraction > .28f ? theme::Mauled : theme::Sinking;
    Rectangle amount = bounds;
    amount.width *= fraction;
    DrawRectangleRec(amount, fill);
    DrawRectangleLinesEx(bounds, 1, theme::RuleSoft);
    for (int i = 1; i < 4; ++i) {
        const float x = bounds.x + bounds.width * i / 4.0f;
        DrawLineEx({x, bounds.y}, {x, bounds.y + bounds.height}, 1, theme::Gutter);
    }
}

void UiPainter::compass(Vector2 centre, float radius, double bearing) const {
    DrawCircleV(centre, radius, ColorAlpha(theme::Panel, .82f));
    DrawCircleLines(static_cast<int>(centre.x), static_cast<int>(centre.y), radius, theme::Rule);
    for (int i = 0; i < 8; ++i) {
        const float angle = i * PI / 4.0f;
        const float inner = radius - (i == 0 ? 8 : i % 2 == 0 ? 5 : 3) * scale_;
        DrawLineEx({centre.x + std::sin(angle) * inner, centre.y - std::cos(angle) * inner},
                   {centre.x + std::sin(angle) * (radius - 2), centre.y - std::cos(angle) * (radius - 2)}, 1,
                   theme::Rule);
    }
    const float dx = std::sin(static_cast<float>(bearing));
    const float dy = -std::cos(static_cast<float>(bearing));
    const float length = radius - 8 * scale_;
    DrawLineEx({centre.x - dx * length, centre.y - dy * length},
               {centre.x + dx * length * .65f, centre.y + dy * length * .65f}, 2.2f * scale_,
               Color{159, 208, 232, 255});
    const Vector2 tip{centre.x + dx * length, centre.y + dy * length};
    DrawTriangle(tip, {tip.x - dx * 8 - dy * 5, tip.y - dy * 8 + dx * 5},
                 {tip.x - dx * 8 + dy * 5, tip.y - dy * 8 - dx * 5}, Color{159, 208, 232, 255});
}

void UiPainter::cornerTitle(std::string_view title, std::string_view subtitle) const {
    text(title, 22, 18, 27, theme::Gold, true);
    text(subtitle, 23, 50, 11, theme::Dim, true);
}

} // namespace broadside::render
