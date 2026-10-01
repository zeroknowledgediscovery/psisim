// PsiSim-owned binding: expose the learned LSM dependency structure.
//
// This file is NOT part of the vendored LSM runtime and is not touched by
// scripts/sync_lsm_runtime.sh. It only uses the public PredictDistribution
// API, so it keeps working across runtime refreshes.
//
// used_columns(trees_dir, tree_ids) returns, for every learned target column
// j, the sorted list of source columns i that the native tree for j actually
// splits on. This is exactly the set used by CenteredLdpPsiState to decide
// which targets receive a centered hard response from source i.

#include "PredictDistribution/PredictDistribution.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <filesystem>
#include <map>
#include <regex>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace py = pybind11;

namespace {

std::vector<int> discover_tree_ids(const fs::path& trees_dir) {
    static const std::regex re(R"(tree_(\d+)\.bin)");
    std::vector<int> ids;
    if (!fs::is_directory(trees_dir)) return ids;
    for (const auto& entry : fs::directory_iterator(trees_dir)) {
        if (!entry.is_regular_file()) continue;
        std::smatch match;
        const std::string name = entry.path().filename().string();
        if (std::regex_match(name, match, re)) {
            ids.push_back(std::stoi(match[1].str()));
        }
    }
    std::sort(ids.begin(), ids.end());
    return ids;
}

std::map<int, std::vector<int>> used_columns_py(
        const std::string& trees_dir,
        std::vector<int> tree_ids) {
    if (tree_ids.empty()) tree_ids = discover_tree_ids(trees_dir);
    if (tree_ids.empty()) {
        throw std::runtime_error("No tree_*.bin found under " + trees_dir);
    }

    std::map<int, std::vector<int>> out;
    {
        py::gil_scoped_release release;
        PredictDistribution predictor(trees_dir);
        for (int target : tree_ids) {
            const std::set<int> used = predictor.usedColumns(target);
            out[target] = std::vector<int>(used.begin(), used.end());
        }
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(psisim_graph, m) {
    m.doc() = "PsiSim: learned LSM dependency structure from native trees";
    m.def("used_columns",
          &used_columns_py,
          py::arg("trees_dir"),
          py::arg("tree_ids") = std::vector<int>{},
          "Return {target: sorted source columns used by tree_target}.");
}
