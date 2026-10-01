#include "PredictDistribution/PredictDistribution.h"
#include "QDistance/qdistance.h"
#include "SourceMaps/SourceMaps.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include <regex>
#include <algorithm>
#include <atomic>
#include <exception>
#include <mutex>
#include <omp.h>

namespace fs = std::filesystem;
namespace py = pybind11;

namespace {
std::vector<int> discover_tree_ids(const fs::path& trees_dir) {
    static const std::regex re(R"(tree_(\d+)\.bin)");
    std::vector<int> ids;
    if (!fs::is_directory(trees_dir)) return ids;
    for (const auto& entry : fs::directory_iterator(trees_dir)) {
        if (!entry.is_regular_file()) continue;
        std::smatch m;
        const std::string name = entry.path().filename().string();
        if (std::regex_match(name, m, re)) ids.push_back(std::stoi(m[1].str()));
    }
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    return ids;
}

std::vector<std::string> row_to_tokens(py::handle row) {
    py::sequence seq = py::reinterpret_borrow<py::sequence>(row);
    const ssize_t p = py::len(seq);
    std::vector<std::string> tokens(static_cast<size_t>(p));
    for (ssize_t j = 0; j < p; ++j) tokens[static_cast<size_t>(j)] = py::str(seq[j]);
    return tokens;
}

std::vector<std::string> omega_tokens(sourcemaps::SourceMapStore& store,
                                      PredictDistribution& predictor,
                                      const std::vector<std::string>& tokens,
                                      const std::vector<int>& codes,
                                      const std::vector<int>& tree_ids) {
    std::vector<std::string> output = tokens;
    for (int tree_id : tree_ids) {
        if (tree_id < 0 || static_cast<size_t>(tree_id) >= output.size()) continue;
        const auto counts = predictor.predict(tree_id, codes);
        if (counts.empty()) continue;
        long long best_count = -1;
        int best_code = counts.begin()->first;
        for (const auto& kv : counts) {
            if (kv.second > best_count) {
                best_count = kv.second;
                best_code = kv.first;
            }
        }
        if (store.tryGet(tree_id)) output[static_cast<size_t>(tree_id)] = store.decodeLabel(tree_id, best_code);
        else output[static_cast<size_t>(tree_id)] = std::to_string(best_code);
    }
    return output;
}
}

static py::dict qdistance_py(const std::string& trees_dir,
                             const std::vector<std::string>& rowA_tokens,
                             const std::vector<std::string>& rowB_tokens,
                             const std::string& run_dir,
                             int cols_per_shard,
                             const std::vector<int>& tree_ids,
                             bool return_per_tree) {
    if (rowA_tokens.size() != rowB_tokens.size()) throw std::invalid_argument("rowA and rowB must have the same length.");
    if (tree_ids.empty()) throw std::invalid_argument("tree_ids must be non-empty.");

    fs::path treesDir = fs::absolute(trees_dir);
    fs::path runDir = run_dir.empty() ? sourcemaps::SourceMapStore::inferRunDir(treesDir) : fs::absolute(run_dir);
    if (runDir.empty() || !fs::exists(runDir / "source_maps")) throw std::runtime_error("run_dir must contain source_maps/");

    sourcemaps::SourceMapStore store(runDir, cols_per_shard);
    std::vector<int> rowA_codes = store.encodeRow(rowA_tokens);
    std::vector<int> rowB_codes = store.encodeRow(rowB_tokens);
    PredictDistribution predictor(treesDir.string());
    predictor.preload(tree_ids);
    const double qd = qdistance(predictor, rowA_codes, rowB_codes, tree_ids);

    py::dict out;
    out["qdistance_bits"] = qd;
    if (return_per_tree) {
        auto per = qdistance_per_tree(predictor, rowA_codes, rowB_codes, tree_ids);
        py::dict d;
        for (auto& kv : per) d[py::int_(kv.first)] = kv.second;
        out["per_tree_bits"] = std::move(d);
    }
    return out;
}

static py::dict predict_tree_probcodes(const std::string& trees_dir,
                                       const std::vector<std::string>& row_tokens,
                                       const std::string& run_dir,
                                       int cols_per_shard,
                                       int tree_id) {
    fs::path treesDir = fs::absolute(trees_dir);
    fs::path runDir = run_dir.empty() ? sourcemaps::SourceMapStore::inferRunDir(treesDir) : fs::absolute(run_dir);
    sourcemaps::SourceMapStore store(runDir, cols_per_shard);
    std::vector<int> codes = store.encodeRow(row_tokens);
    PredictDistribution pred(treesDir.string());
    const std::map<int, int> counts = pred.predict(tree_id, codes);
    long long tot = 0;
    for (auto& kv : counts) tot += kv.second;
    py::dict out;
    if (tot > 0) for (auto& kv : counts) out[py::int_(kv.first)] = double(kv.second) / double(tot);
    return out;
}

static py::array_t<double> qdistance_matrix_py(const std::string& trees_dir,
                                               py::object rows,
                                               const std::string& run_dir,
                                               int cols_per_shard,
                                               std::vector<int> tree_ids,
                                               int threads) {
    fs::path treesDir = fs::absolute(trees_dir);
    fs::path runDir = run_dir.empty() ? sourcemaps::SourceMapStore::inferRunDir(treesDir) : fs::absolute(run_dir);
    if (runDir.empty() || !fs::exists(runDir / "source_maps")) throw std::runtime_error("run_dir must contain source_maps/");
    if (tree_ids.empty()) tree_ids = discover_tree_ids(treesDir);
    if (tree_ids.empty()) throw std::runtime_error("no trees found");

    py::sequence rows_seq = py::reinterpret_borrow<py::sequence>(rows);
    const ssize_t num_rows = py::len(rows_seq);
    if (num_rows <= 0) throw std::invalid_argument("rows must be a non-empty sequence");

    sourcemaps::SourceMapStore store(runDir, cols_per_shard);
    std::vector<std::vector<int>> codes(static_cast<size_t>(num_rows));
    size_t p = 0;
    for (ssize_t i = 0; i < num_rows; ++i) {
        auto tokens = row_to_tokens(rows_seq[i]);
        if (i == 0) p = tokens.size();
        else if (tokens.size() != p) throw std::invalid_argument("all rows must have the same number of columns");
        codes[static_cast<size_t>(i)] = store.encodeRow(tokens);
    }

    PredictDistribution predictor(treesDir.string());
    predictor.preload(tree_ids);

    auto D = py::array_t<double>({num_rows, num_rows});
    auto d = D.mutable_unchecked<2>();
    const int requested_threads = threads > 0 ? threads : omp_get_max_threads();
    if (requested_threads < 1) throw std::invalid_argument("threads must be >= 1, or 0 for OpenMP default");

    std::atomic<bool> failed{false};
    std::exception_ptr first_error;
    std::mutex error_mutex;

    {
        py::gil_scoped_release release;
        #pragma omp parallel for schedule(dynamic) num_threads(requested_threads)
        for (ssize_t i = 0; i < num_rows; ++i) {
            if (failed.load(std::memory_order_relaxed)) continue;
            try {
                d(i, i) = 0.0;
                for (ssize_t j = i + 1; j < num_rows; ++j) {
                    if (failed.load(std::memory_order_relaxed)) break;
                    const double v = qdistance(predictor,
                                               codes[static_cast<size_t>(i)],
                                               codes[static_cast<size_t>(j)],
                                               tree_ids);
                    d(i, j) = v;
                    d(j, i) = v;
                }
            } catch (...) {
                failed.store(true, std::memory_order_relaxed);
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!first_error) first_error = std::current_exception();
            }
        }
    }

    if (first_error) std::rethrow_exception(first_error);
    return D;
}

static py::array_t<double> instability_batch_py(const std::string& trees_dir, py::object rows, const std::string& run_dir, int cols_per_shard, std::vector<int> tree_ids, int threads) {
    fs::path treesDir = fs::absolute(trees_dir);
    fs::path runDir = run_dir.empty() ? sourcemaps::SourceMapStore::inferRunDir(treesDir) : fs::absolute(run_dir);
    if (runDir.empty() || !fs::exists(runDir / "source_maps")) throw std::runtime_error("run_dir must contain source_maps/");
    if (tree_ids.empty()) tree_ids = discover_tree_ids(treesDir);
    if (tree_ids.empty()) throw std::runtime_error("no trees found");
    py::sequence seq = py::reinterpret_borrow<py::sequence>(rows);
    const ssize_t n = py::len(seq);
    if (n <= 0) throw std::invalid_argument("rows must be non-empty");
    sourcemaps::SourceMapStore store(runDir, cols_per_shard);
    std::vector<std::vector<int>> codes(static_cast<size_t>(n));
    for (ssize_t i = 0; i < n; ++i) codes[static_cast<size_t>(i)] = store.encodeRow(row_to_tokens(seq[i]));
    PredictDistribution predictor(treesDir.string());
    predictor.preload(tree_ids);
    auto out = py::array_t<double>(n);
    auto o = out.mutable_unchecked<1>();
    const int nt = threads > 0 ? threads : omp_get_max_threads();
    std::atomic<bool> failed{false};
    std::exception_ptr first_error;
    std::mutex error_mutex;
    {
        py::gil_scoped_release release;
        #pragma omp parallel for schedule(dynamic) num_threads(nt)
        for (ssize_t i = 0; i < n; ++i) {
            if (failed.load(std::memory_order_relaxed)) continue;
            try {
                const auto& row_codes = codes[static_cast<size_t>(i)];
                std::vector<int> omega_codes = row_codes;
                for (int tid : tree_ids) {
                    if (tid < 0 || static_cast<size_t>(tid) >= omega_codes.size()) continue;
                    const auto counts = predictor.predict(tid, row_codes);
                    if (counts.empty()) continue;
                    auto best = std::max_element(counts.begin(), counts.end(),
                        [](const auto& a, const auto& b) { return a.second < b.second; });
                    omega_codes[static_cast<size_t>(tid)] = best->first;
                }
                o(i) = qdistance(predictor, row_codes, omega_codes, tree_ids);
            } catch (...) {
                failed.store(true, std::memory_order_relaxed);
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!first_error) first_error = std::current_exception();
            }
        }
    }
    if (first_error) std::rethrow_exception(first_error);
    return out;
}

static py::dict instability_py(const std::string& trees_dir,
                               py::object row,
                               const std::string& run_dir,
                               int cols_per_shard,
                               std::vector<int> tree_ids) {
    fs::path treesDir = fs::absolute(trees_dir);
    fs::path runDir = run_dir.empty() ? sourcemaps::SourceMapStore::inferRunDir(treesDir) : fs::absolute(run_dir);
    if (runDir.empty() || !fs::exists(runDir / "source_maps")) throw std::runtime_error("run_dir must contain source_maps/");
    if (tree_ids.empty()) tree_ids = discover_tree_ids(treesDir);
    if (tree_ids.empty()) throw std::runtime_error("no trees found");

    auto tokens = row_to_tokens(row);
    sourcemaps::SourceMapStore store(runDir, cols_per_shard);
    auto codes = store.encodeRow(tokens);
    PredictDistribution predictor(treesDir.string());
    predictor.preload(tree_ids);
    auto omega = omega_tokens(store, predictor, tokens, codes, tree_ids);
    auto omega_codes = store.encodeRow(omega);
    const double v = qdistance(predictor, codes, omega_codes, tree_ids);

    py::dict out;
    out["instability"] = v;
    py::list omega_list;
    for (const auto& s : omega) omega_list.append(py::str(s));
    out["omega"] = std::move(omega_list);
    return out;
}

PYBIND11_MODULE(qdistance, m) {
    m.doc() = "qdistance: average Jensen-Shannon divergence between two rows' predicted distributions";
    m.def("qdistance", &qdistance_py,
          py::arg("trees_dir"), py::arg("rowA_tokens"), py::arg("rowB_tokens"),
          py::arg("run_dir") = "", py::arg("cols_per_shard") = 50000,
          py::arg("tree_ids"), py::arg("return_per_tree") = false);
    m.def("predict_tree_probcodes", &predict_tree_probcodes,
          py::arg("trees_dir"), py::arg("row_tokens"), py::arg("run_dir") = "",
          py::arg("cols_per_shard") = 50000, py::arg("tree_id"));
    m.def("qdistance_matrix", &qdistance_matrix_py,
          py::arg("trees_dir"), py::arg("rows"), py::arg("run_dir") = "",
          py::arg("cols_per_shard") = 50000, py::arg("tree_ids") = std::vector<int>{},
          py::arg("threads") = 0);
    m.def("instability_batch", &instability_batch_py,
          py::arg("trees_dir"), py::arg("rows"), py::arg("run_dir") = "",
          py::arg("cols_per_shard") = 50000, py::arg("tree_ids") = std::vector<int>{},
          py::arg("threads") = 0);
    m.def("instability", &instability_py,
          py::arg("trees_dir"), py::arg("row"), py::arg("run_dir") = "",
          py::arg("cols_per_shard") = 50000, py::arg("tree_ids") = std::vector<int>{});
}
