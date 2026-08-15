#include "reference.h"
#include <cmath>
#include <fstream>
#include <regex>
#include <stdexcept>

namespace broadside::reference {
std::vector<SemanticRow> load(const std::filesystem::path &path) {
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("cannot open semantic reference: " + path.string());
    const std::regex pattern(
        R"(^([0-4])\s+(\w+)\s+(\w+)\s+(\d+)\s+w=([01-])\s+t=([0-9.]+)\s+s=([0-9.]+)/([0-9.]+)$)");
    std::vector<SemanticRow> rows;
    std::string line;
    int number = 0;
    while (std::getline(input, line)) {
        ++number;
        if (line.empty() || line.front() == '#')
            continue;
        std::smatch match;
        if (!std::regex_match(line, match, pattern))
            throw std::runtime_error("malformed semantic reference line " + std::to_string(number));
        SemanticRow row;
        row.hull = std::stoi(match[1]);
        row.left = match[2];
        row.right = match[3];
        row.seed = static_cast<unsigned>(std::stoul(match[4]));
        row.winner = match[5] == "-" ? -1 : std::stoi(match[5]);
        row.seconds = std::stod(match[6]);
        row.leftStructure = std::stod(match[7]);
        row.rightStructure = std::stod(match[8]);
        rows.push_back(std::move(row));
    }
    if (rows.size() != 900)
        throw std::runtime_error("semantic reference must contain exactly 900 battles");
    return rows;
}
Comparison compare(const std::vector<SemanticRow> &expected, const std::vector<SemanticRow> &actual,
                   double tickTolerance, double structureTolerance) {
    if (expected.size() != actual.size())
        throw std::invalid_argument("semantic comparison row counts differ");
    Comparison result;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const auto &e = expected[i];
        const auto &a = actual[i];
        if (e.hull != a.hull || e.left != a.left || e.right != a.right || e.seed != a.seed ||
            e.winner != a.winner)
            ++result.winnerMismatches;
        if (std::abs(e.seconds - a.seconds) > tickTolerance + .0005f)
            ++result.timeMismatches;
        if (std::abs(e.leftStructure - a.leftStructure) > structureTolerance ||
            std::abs(e.rightStructure - a.rightStructure) > structureTolerance)
            ++result.structureMismatches;
    }
    return result;
}
} // namespace broadside::reference
