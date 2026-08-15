#include "types.h"
#include <algorithm>

namespace broadside::sim {
namespace {
const std::vector<PartDef> Parts = {{PartId::Timber, "Hull timber", '=', 1, 9},
                                    {PartId::Heavy, "Heavy timbers", '#', 3, 30, 0, 0, 2},
                                    {PartId::Crew, "Crew quarters", 'c', 5, 14, 3},
                                    {PartId::Mast, "Mast", '^', 4, 11, 0, 1},
                                    {PartId::Magazine, "Powder magazine", '*', 4, 8, 0, 0, 0, true},
                                    {PartId::Swivel,
                                     "Swivel gun",
                                     'o',
                                     4,
                                     11,
                                     0,
                                     1,
                                     0,
                                     false,
                                     false,
                                     false,
                                     true,
                                     {Arc::All, 180, 26, 1, 1, 5, 26, 3, 1, 1, false}},
                                    {PartId::GunDeck,
                                     "Gun deck",
                                     'G',
                                     8,
                                     17,
                                     0,
                                     2,
                                     0,
                                     false,
                                     false,
                                     false,
                                     true,
                                     {Arc::Side, 50, 38, 1.8f, 3, 7, 30, 4, 1, 1, false}},
                                    {PartId::Carronade,
                                     "Carronade",
                                     'K',
                                     9,
                                     14,
                                     0,
                                     1,
                                     0,
                                     false,
                                     false,
                                     false,
                                     true,
                                     {Arc::Side, 55, 24, 1.9f, 2, 6, 28, 9, 2, 2, false}},
                                    {PartId::LongGun,
                                     "Long gun",
                                     'L',
                                     9,
                                     14,
                                     0,
                                     2,
                                     0,
                                     false,
                                     false,
                                     true,
                                     true,
                                     {Arc::Bow, 32, 48, 2.2f, 1, 3, 39, 22, 2, 2, true}},
                                    {PartId::Helm, "Helm", '@', 0, 20, 0, 0, 0, false, true}};
const std::vector<PartId> Buyable = {PartId::Timber,  PartId::Heavy,     PartId::Crew,
                                     PartId::Mast,    PartId::Magazine,  PartId::Swivel,
                                     PartId::GunDeck, PartId::Carronade, PartId::LongGun};
const char *const art[] = {".#.",   "###",   "###",   "###",   ".#.",   ".#.",   "###",   "###",
                           "###",   "###",   "###",   ".#.",   "..#..", ".###.", "#####", "#####",
                           "#####", ".###.", "..#..", "..#..", ".###.", ".###.", "#####", "#####",
                           "#####", ".###.", ".###.", "..#..", "..#..", ".###.", "#####", "#####",
                           "#####", "#####", "#####", "#####", ".###.", "..#.."};
const int artRows[] = {5, 7, 7, 9, 10};
const int artStarts[] = {0, 5, 12, 19, 28};
} // namespace
const PartDef &part(PartId id) {
    return Parts[static_cast<int>(id)];
}
const std::vector<PartId> &buyableParts() {
    return Buyable;
}
const std::vector<HullDef> &hulls() {
    static const std::vector<HullDef> result = [] {
        std::vector<HullDef> out;
        const char *names[] = {"Sloop", "Brig", "Frigate", "Heavy frigate", "Ship of the line"};
        const int widths[] = {3, 3, 5, 5, 5};
        for (int h = 0; h < 5; ++h) {
            HullDef d;
            d.name = names[h];
            d.width = widths[h];
            d.length = artRows[h];
            const int centre = (d.length - 1) / 2;
            d.bowZ = -centre;
            d.bowLimit = d.bowZ + 2;
            for (int row = 0; row < artRows[h]; ++row)
                for (int col = 0; col < d.width; ++col) {
                    if (art[artStarts[h] + row][col] == '#')
                        d.cells.push_back({col - (d.width - 1) / 2, row - centre});
                }
            out.push_back(std::move(d));
        }
        return out;
    }();
    return result;
}
const HullDef &hull(int index) {
    return hulls().at(static_cast<std::size_t>(index));
}
void setPart(Design &d, Coord c, Slot slot) {
    if (!d.parts.contains(c))
        d.order.push_back(c);
    d.parts[c] = slot;
}
void erasePart(Design &d, Coord c) {
    d.parts.erase(c);
    d.order.erase(std::remove(d.order.begin(), d.order.end(), c), d.order.end());
}
Design makeDesign() {
    Design d;
    setPart(d, {0, 0}, {PartId::Helm, part(PartId::Helm).hp});
    return d;
}
Design cloneDesign(const Design &d) {
    return d;
}
Coord coord(int dx, int dz) {
    return {dx, dz};
}
bool isHullCell(int i, Coord c) {
    const auto &cells = hull(i).cells;
    return std::find(cells.begin(), cells.end(), c) != cells.end();
}
bool isBowCell(int i, int dz) {
    return dz <= hull(i).bowLimit;
}
int hullCellCount(int i) {
    return static_cast<int>(hull(i).cells.size());
}
void fitDesignToHull(Design &d, int i) {
    std::vector<Coord> remove;
    for (const auto &[c, _] : d.parts)
        if (!isHullCell(i, c))
            remove.push_back(c);
    for (auto c : remove)
        erasePart(d, c);
    if (!d.parts.contains({0, 0}))
        setPart(d, {0, 0}, {PartId::Helm, part(PartId::Helm).hp});
}
} // namespace broadside::sim
