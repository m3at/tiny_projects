#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace broadside::reference {
struct SemanticRow {
    int hull = 0;
    std::string left, right;
    unsigned seed = 0;
    int winner = -1;
    double seconds = 0, leftStructure = 0, rightStructure = 0;
};
struct Comparison {
    int winnerMismatches = 0, timeMismatches = 0, structureMismatches = 0;
    bool ok() const {
        return winnerMismatches == 0 && timeMismatches == 0 && structureMismatches == 0;
    }
};
std::vector<SemanticRow> load(const std::filesystem::path &path);
Comparison compare(const std::vector<SemanticRow> &expected, const std::vector<SemanticRow> &actual,
                   double tickTolerance = 1.0 / 60.0, double structureTolerance = .001);
} // namespace broadside::reference
