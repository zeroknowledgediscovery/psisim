#include "PredictDistribution/PredictDistribution.h"
#include <nlohmann/json.hpp>
#include "SourceMaps/SourceMaps.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <filesystem>
#include <map>
#include <set>
#include <regex>
#include <random>
#include <numeric>
#include <cstdint>
#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <shared_mutex>
#include <iostream>
#include <chrono>
#include <atomic>
#include <exception>
#include <mutex>
#include <omp.h>

namespace fs = std::filesystem;
namespace py = pybind11;

namespace {

static std::vector<int> encode_row_codes(sourcemaps::SourceMapStore& store,
                                         py::array row,
                                         bool raw) {
    std::vector<int> row_codes;
    if (raw) {
        if (row.ndim() != 1) {
            throw std::runtime_error("raw=True expects a 1-D numpy array");
        }
        py::sequence seq = py::reinterpret_borrow<py::sequence>(row);
        const ssize_t n = py::len(seq);
        std::vector<std::string> tokens(static_cast<size_t>(n));
        for (ssize_t i = 0; i < n; ++i) {
            tokens[static_cast<size_t>(i)] = py::str(seq[i]);
        }
        row_codes = store.encodeRow(tokens);
    } else {
        auto a = row.cast<py::array_t<long long, py::array::c_style | py::array::forcecast>>();
        if (a.ndim() != 1) {
            throw std::runtime_error("raw=False expects a 1-D integer array");
        }
        auto r = a.unchecked<1>();
        row_codes.resize(static_cast<size_t>(r.shape(0)));
        for (ssize_t i = 0; i < r.shape(0); ++i) {
            row_codes[static_cast<size_t>(i)] = static_cast<int>(r(i));
        }
    }
    return row_codes;
}

static py::dict counts_to_prob_dict(sourcemaps::SourceMapStore& store,
                                    int target_col_id,
                                    const std::map<int, int>& counts) {
    if (!store.tryGet(target_col_id)) {
        long total = 0;
        for (const auto& kv : counts) total += kv.second;
        py::dict out;
        for (const auto& kv : counts) {
            double p = total ? double(kv.second) / double(total) : 0.0;
            out[py::cast(kv.first)] = p;
        }
        return out;
    }

    std::map<std::string, long> labeled;
    long total = 0;
    for (const auto& kv : counts) {
        const std::string label = store.decodeLabel(target_col_id, kv.first);
        labeled[label] += kv.second;
        total += kv.second;
    }

    py::dict out;
    for (const auto& kv : labeled) {
        double p = total ? double(kv.second) / double(total) : 0.0;
        out[py::str(kv.first)] = p;
    }
    return out;
}

static std::vector<int> discover_tree_ids(const fs::path& trees_dir) {
    static const std::regex re(R"(tree_(\d+)\.bin)");
    std::vector<int> ids;
    if (!fs::is_directory(trees_dir)) return ids;
    for (const auto& entry : fs::directory_iterator(trees_dir)) {
        if (!entry.is_regular_file()) continue;
        std::smatch m;
        const std::string name = entry.path().filename().string();
        if (std::regex_match(name, m, re)) {
            ids.push_back(std::stoi(m[1].str()));
        }
    }
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    return ids;
}

static std::string argmax_label(sourcemaps::SourceMapStore& store,
                                int target_col_id,
                                const std::map<int, int>& counts) {
    if (counts.empty()) return "";
    long long best_count = -1;
    int best_code = counts.begin()->first;
    for (const auto& kv : counts) {
        if (kv.second > best_count) {
            best_count = kv.second;
            best_code = kv.first;
        }
    }
    if (store.tryGet(target_col_id)) return store.decodeLabel(target_col_id, best_code);
    return std::to_string(best_code);
}

static double token_log2_prob(sourcemaps::SourceMapStore& store,
                             int target_col_id,
                             const std::map<int, int>& counts,
                             const std::string& token) {
    long long total = 0;
    for (const auto& kv : counts) total += kv.second;
    if (total <= 0) return -std::numeric_limits<double>::infinity();

    long long hit = 0;
    if (store.tryGet(target_col_id)) {
        for (const auto& kv : counts) {
            if (store.decodeLabel(target_col_id, kv.first) == token) hit += kv.second;
        }
    } else {
        try {
            int code = std::stoi(token);
            auto it = counts.find(code);
            if (it != counts.end()) hit = it->second;
        } catch (...) {
            hit = 0;
        }
    }
    if (hit <= 0) return -std::numeric_limits<double>::infinity();
    return std::log2(static_cast<double>(hit) / static_cast<double>(total));
}


static ProbabilityDistribution counts_to_distribution(
        const std::map<int, int>& counts) {
    double total = 0.0;
    for (const auto& kv : counts) total += static_cast<double>(kv.second);
    if (!(total > 0.0)) throw std::runtime_error("empty predicted distribution");
    ProbabilityDistribution out;
    for (const auto& kv : counts) {
        out[kv.first] = static_cast<double>(kv.second) / total;
    }
    return out;
}

static ProbabilityDistribution python_to_distribution(
        sourcemaps::SourceMapStore& store,
        int col_id,
        py::handle obj,
        bool raw) {
    if (!py::isinstance<py::dict>(obj)) {
        throw std::invalid_argument("each Psi coordinate must be a dict");
    }
    py::dict d = py::reinterpret_borrow<py::dict>(obj);
    ProbabilityDistribution out;
    double total = 0.0;
    for (auto item : d) {
        const double p = py::cast<double>(item.second);
        if (!std::isfinite(p) || p < 0.0) {
            throw std::invalid_argument("Psi probabilities must be finite and non-negative");
        }
        int code;
        if (raw) {
            code = store.encodeToken(col_id, py::str(item.first));
            if (code == 0) {
                throw std::invalid_argument(
                    "Psi contains an unknown label in column " + std::to_string(col_id));
            }
        } else {
            code = py::cast<int>(item.first);
        }
        out[code] += p;
        total += p;
    }
    if (!(total > 0.0)) {
        throw std::invalid_argument("every Psi coordinate must have positive total mass");
    }
    for (auto& kv : out) kv.second /= total;
    return out;
}

static ProbabilityRow python_to_psi(
        sourcemaps::SourceMapStore& store,
        py::object psi,
        bool raw) {
    py::sequence seq = py::reinterpret_borrow<py::sequence>(psi);
    const ssize_t n = py::len(seq);
    if (n <= 0) throw std::invalid_argument("Psi must be non-empty");
    ProbabilityRow out(static_cast<size_t>(n));
    for (ssize_t i = 0; i < n; ++i) {
        out[static_cast<size_t>(i)] =
            python_to_distribution(store, static_cast<int>(i), seq[i], raw);
    }
    return out;
}

static py::dict distribution_to_python(
        sourcemaps::SourceMapStore& store,
        int col_id,
        const ProbabilityDistribution& q,
        bool raw) {
    py::dict out;
    for (const auto& kv : q) {
        if (raw) {
            out[py::str(store.decodeLabel(col_id, kv.first))] = kv.second;
        } else {
            out[py::int_(kv.first)] = kv.second;
        }
    }
    return out;
}

static py::list psi_to_python(
        sourcemaps::SourceMapStore& store,
        const ProbabilityRow& psi,
        bool raw) {
    py::list out(psi.size());
    for (size_t i = 0; i < psi.size(); ++i) {
        out[static_cast<ssize_t>(i)] =
            distribution_to_python(store, static_cast<int>(i), psi[i], raw);
    }
    return out;
}

}  // namespace

static py::dict predict_distribution_py(const std::string& trees_dir,
                                        int target_col_id,
                                        py::array row,
                                        bool raw,
                                        std::string run_dir,
                                        int cols_per_shard) {
    fs::path treesDir = fs::absolute(trees_dir);
    fs::path runDir = run_dir.empty()
                          ? sourcemaps::SourceMapStore::inferRunDir(treesDir)
                          : fs::absolute(run_dir);

    if (raw && (runDir.empty() || !fs::exists(runDir / "source_maps"))) {
        throw std::runtime_error("When raw=True, run_dir must contain source_maps/");
    }

    sourcemaps::SourceMapStore store(runDir, cols_per_shard);
    const std::vector<int> row_codes = encode_row_codes(store, row, raw);

    PredictDistribution pd(treesDir.string());
    auto counts = pd.predict(target_col_id, row_codes);
    return counts_to_prob_dict(store, target_col_id, counts);
}

static py::list predict_distributions_py(const std::string& trees_dir,
                                         py::array row,
                                         bool raw,
                                         std::string run_dir,
                                         int cols_per_shard,
                                         std::vector<int> tree_ids) {
    fs::path treesDir = fs::absolute(trees_dir);
    fs::path runDir = run_dir.empty()
                          ? sourcemaps::SourceMapStore::inferRunDir(treesDir)
                          : fs::absolute(run_dir);

    if (raw && (runDir.empty() || !fs::exists(runDir / "source_maps"))) {
        throw std::runtime_error("When raw=True, run_dir must contain source_maps/");
    }

    if (tree_ids.empty()) {
        tree_ids = discover_tree_ids(treesDir);
    }
    if (tree_ids.empty()) {
        throw std::runtime_error("No tree_*.bin found and tree_ids is empty");
    }

    sourcemaps::SourceMapStore store(runDir, cols_per_shard);
    const std::vector<int> row_codes = encode_row_codes(store, row, raw);

    PredictDistribution pd(treesDir.string());

    const int n = *std::max_element(tree_ids.begin(), tree_ids.end()) + 1;
    py::list out(n);
    for (int i = 0; i < n; ++i) out[i] = py::none();

    for (int tid : tree_ids) {
        auto counts = pd.predict(tid, row_codes);
        out[tid] = counts_to_prob_dict(store, tid, counts);
    }
    return out;
}


static py::list row_to_psi_py(
        const std::string& trees_dir,
        py::array row,
        bool raw,
        std::string run_dir,
        int cols_per_shard,
        std::vector<int> tree_ids) {
    fs::path treesDir = fs::absolute(trees_dir);
    fs::path runDir = run_dir.empty()
                          ? sourcemaps::SourceMapStore::inferRunDir(treesDir)
                          : fs::absolute(run_dir);
    if (raw && (runDir.empty() || !fs::exists(runDir / "source_maps"))) {
        throw std::runtime_error("When raw=True, run_dir must contain source_maps/");
    }
    if (tree_ids.empty()) tree_ids = discover_tree_ids(treesDir);
    if (tree_ids.empty()) throw std::runtime_error("No tree_*.bin found");

    sourcemaps::SourceMapStore store(runDir, cols_per_shard);
    const std::vector<int> row_codes = encode_row_codes(store, row, raw);
    std::unordered_set<int> available(tree_ids.begin(), tree_ids.end());

    PredictDistribution predictor(treesDir.string());
    predictor.preload(tree_ids);

    ProbabilityRow psi(row_codes.size());
    for (size_t i = 0; i < row_codes.size(); ++i) {
        const int code = row_codes[i];
        if (code != 0) {
            psi[i][code] = 1.0;
            continue;
        }
        if (!available.count(static_cast<int>(i))) {
            throw std::runtime_error(
                "missing column has no learned tree: " + std::to_string(i));
        }
        psi[i] = counts_to_distribution(
            predictor.predict(static_cast<int>(i), row_codes));
    }
    return psi_to_python(store, psi, raw);
}

static py::dict predict_distribution_soft_py(
        const std::string& trees_dir,
        int target_col_id,
        py::object psi_obj,
        bool raw,
        std::string run_dir,
        int cols_per_shard) {
    fs::path treesDir = fs::absolute(trees_dir);
    fs::path runDir = run_dir.empty()
                          ? sourcemaps::SourceMapStore::inferRunDir(treesDir)
                          : fs::absolute(run_dir);
    if (raw && (runDir.empty() || !fs::exists(runDir / "source_maps"))) {
        throw std::runtime_error("When raw=True, run_dir must contain source_maps/");
    }

    sourcemaps::SourceMapStore store(runDir, cols_per_shard);
    const ProbabilityRow psi = python_to_psi(store, psi_obj, raw);
    PredictDistribution predictor(treesDir.string());
    const ProbabilityDistribution result =
        predictor.predictSoft(target_col_id, psi);
    return distribution_to_python(store, target_col_id, result, raw);
}



static ProbabilityDistribution normalize_probability_distribution(
        ProbabilityDistribution q) {
    double total = 0.0;
    for (const auto& kv : q) {
        if (!std::isfinite(kv.second) || kv.second < 0.0) {
            throw std::invalid_argument(
                "probabilities must be finite and non-negative");
        }
        total += kv.second;
    }
    if (!(total > 0.0)) {
        throw std::invalid_argument("probability distribution has zero mass");
    }
    for (auto& kv : q) kv.second /= total;
    return q;
}

static ProbabilityDistribution multinomial_empirical(
        const ProbabilityDistribution& p,
        int n,
        uint64_t seed) {
    if (n <= 0) throw std::invalid_argument("n must be > 0");
    std::vector<int> codes;
    std::vector<double> weights;
    codes.reserve(p.size());
    weights.reserve(p.size());
    for (const auto& kv : p) {
        if (!std::isfinite(kv.second) || kv.second < 0.0) {
            throw std::invalid_argument(
                "multinomial_empirical: invalid probability");
        }
        if (kv.second > 0.0) {
            codes.push_back(kv.first);
            weights.push_back(kv.second);
        }
    }
    if (codes.empty()) {
        throw std::invalid_argument(
            "multinomial_empirical: empty positive support");
    }

    std::mt19937_64 rng(seed ? seed : 1ULL);
    std::discrete_distribution<size_t> draw(weights.begin(), weights.end());
    std::map<int, int> counts;
    for (int r = 0; r < n; ++r) {
        counts[codes[draw(rng)]] += 1;
    }

    ProbabilityDistribution out;
    for (const auto& kv : p) out[kv.first] = 0.0;
    for (const auto& kv : counts) {
        out[kv.first] = static_cast<double>(kv.second) /
                        static_cast<double>(n);
    }
    return out;
}


static ProbabilityDistribution multinomial_mode_empirical(
        const ProbabilityDistribution& p,
        int n) {
    if (n <= 0) throw std::invalid_argument("n must be > 0");

    ProbabilityDistribution out;
    std::map<int, int> counts;
    for (const auto& kv : p) {
        if (!std::isfinite(kv.second) || kv.second < 0.0) {
            throw std::invalid_argument(
                "multinomial_mode_empirical: invalid probability");
        }
        out[kv.first] = 0.0;
        counts[kv.first] = 0;
    }

    // The multinomial log-probability is separable concave in the counts:
    // log P(c) = const + sum_i[c_i log p_i - log(c_i!)].
    // Greedily assigning each next count to the largest p_i/(c_i+1)
    // therefore returns a global mode under sum_i c_i=n.
    for (int r = 0; r < n; ++r) {
        int best_code = 0;
        double best_score = -1.0;
        bool found = false;
        for (const auto& kv : p) {
            if (!(kv.second > 0.0)) continue;
            const double score =
                kv.second /
                static_cast<double>(counts[kv.first] + 1);
            if (!found || score > best_score) {
                found = true;
                best_score = score;
                best_code = kv.first;
            }
        }
        if (!found) {
            throw std::invalid_argument(
                "multinomial_mode_empirical: empty positive support");
        }
        counts[best_code] += 1;
    }

    for (const auto& kv : counts) {
        out[kv.first] =
            static_cast<double>(kv.second) /
            static_cast<double>(n);
    }
    return out;
}

static double kl_divergence(
        const ProbabilityDistribution& q,
        const ProbabilityDistribution& p) {
    double d = 0.0;
    for (const auto& kv : q) {
        const double qv = kv.second;
        if (!(qv > 0.0)) continue;
        const auto it = p.find(kv.first);
        if (it == p.end() || !(it->second > 0.0)) {
            return std::numeric_limits<double>::infinity();
        }
        d += qv * std::log(qv / it->second);
    }
    return d;
}

static ProbabilityDistribution simplex_project(
        const ProbabilityDistribution& q,
        bool* changed = nullptr) {
    if (q.empty()) throw std::invalid_argument("cannot project empty distribution");

    std::vector<std::pair<int, double>> items(q.begin(), q.end());
    std::vector<double> u;
    u.reserve(items.size());
    bool needs = false;
    double sum = 0.0;
    for (const auto& kv : items) {
        if (!std::isfinite(kv.second)) {
            throw std::invalid_argument("non-finite value in probability update");
        }
        u.push_back(kv.second);
        sum += kv.second;
        if (kv.second < -1e-14 || kv.second > 1.0 + 1e-14) needs = true;
    }
    if (std::abs(sum - 1.0) > 1e-12) needs = true;

    if (!needs) {
        ProbabilityDistribution out = q;
        for (auto& kv : out) {
            if (kv.second < 0.0 && kv.second > -1e-14) kv.second = 0.0;
        }
        double z = 0.0;
        for (const auto& kv : out) z += kv.second;
        for (auto& kv : out) kv.second /= z;
        if (changed) *changed = false;
        return out;
    }

    std::sort(u.begin(), u.end(), std::greater<double>());
    double cssv = 0.0;
    int rho = -1;
    for (size_t j = 0; j < u.size(); ++j) {
        cssv += u[j];
        const double theta = (cssv - 1.0) /
                             static_cast<double>(j + 1);
        if (u[j] - theta > 0.0) rho = static_cast<int>(j);
    }
    if (rho < 0) {
        throw std::logic_error("simplex projection failed");
    }
    cssv = 0.0;
    for (int j = 0; j <= rho; ++j) cssv += u[static_cast<size_t>(j)];
    const double theta = (cssv - 1.0) /
                         static_cast<double>(rho + 1);

    ProbabilityDistribution out;
    for (const auto& kv : items) {
        out[kv.first] = std::max(0.0, kv.second - theta);
    }
    double z = 0.0;
    for (const auto& kv : out) z += kv.second;
    if (!(z > 0.0)) throw std::logic_error("simplex projection returned zero mass");
    for (auto& kv : out) kv.second /= z;
    if (changed) *changed = true;
    return out;
}

static ProbabilityDistribution add_response_delta(
        const ProbabilityDistribution& current,
        const ProbabilityDistribution& base,
        const ProbabilityDistribution& perturbed,
        double response_scale,
        bool* projected) {
    std::set<int> keys;
    for (const auto& kv : current) keys.insert(kv.first);
    for (const auto& kv : base) keys.insert(kv.first);
    for (const auto& kv : perturbed) keys.insert(kv.first);

    ProbabilityDistribution candidate;
    for (int code : keys) {
        const double cur = current.count(code) ? current.at(code) : 0.0;
        const double b = base.count(code) ? base.at(code) : 0.0;
        const double p = perturbed.count(code) ? perturbed.at(code) : 0.0;
        candidate[code] = cur + response_scale * (p - b);
    }
    return simplex_project(candidate, projected);
}

static ProbabilityRow phi_native(
        PredictDistribution& predictor,
        const ProbabilityRow& psi,
        const std::vector<int>& tree_ids,
        int threads) {
    ProbabilityRow next = psi;
    const int nt = threads > 0 ? threads : omp_get_max_threads();
    std::atomic<bool> failed{false};
    std::exception_ptr first_error;
    std::mutex error_mutex;

    #pragma omp parallel for schedule(dynamic) num_threads(nt)
    for (ssize_t k = 0; k < static_cast<ssize_t>(tree_ids.size()); ++k) {
        if (failed.load(std::memory_order_relaxed)) continue;
        try {
            const int tid = tree_ids[static_cast<size_t>(k)];
            if (tid < 0 || static_cast<size_t>(tid) >= psi.size()) continue;
            next[static_cast<size_t>(tid)] = predictor.predictSoft(tid, psi);
        } catch (...) {
            failed.store(true, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lock(error_mutex);
            if (!first_error) first_error = std::current_exception();
        }
    }
    if (first_error) std::rethrow_exception(first_error);
    return next;
}

static double tv_distance(const ProbabilityDistribution& a,
                          const ProbabilityDistribution& b) {
    auto ia = a.begin();
    auto ib = b.begin();
    double l1 = 0.0;
    while (ia != a.end() || ib != b.end()) {
        if (ib == b.end() || (ia != a.end() && ia->first < ib->first)) {
            l1 += std::abs(ia->second);
            ++ia;
        } else if (ia == a.end() || ib->first < ia->first) {
            l1 += std::abs(ib->second);
            ++ib;
        } else {
            l1 += std::abs(ia->second - ib->second);
            ++ia;
            ++ib;
        }
    }
    return 0.5 * l1;
}

struct PsiStepSummary {
    int step = 0;
    double mean_tv = 0.0;
    double max_tv = 0.0;
    int max_col = -1;
};

static PsiStepSummary summarize_step(
        const ProbabilityRow& before,
        const ProbabilityRow& after,
        const std::vector<int>& tree_ids,
        int step) {
    PsiStepSummary out;
    out.step = step;
    double total = 0.0;
    int used = 0;
    for (int tid : tree_ids) {
        if (tid < 0 ||
            static_cast<size_t>(tid) >= before.size() ||
            static_cast<size_t>(tid) >= after.size()) continue;
        const double d = tv_distance(
            before[static_cast<size_t>(tid)],
            after[static_cast<size_t>(tid)]);
        total += d;
        ++used;
        if (d > out.max_tv) {
            out.max_tv = d;
            out.max_col = tid;
        }
    }
    out.mean_tv = used ? total / static_cast<double>(used) : 0.0;
    return out;
}

class PsiState {
public:
    PsiState(const std::string& trees_dir,
             ProbabilityRow initial,
             bool raw,
             std::string run_dir,
             int cols_per_shard,
             std::vector<int> tree_ids)
        : treesDir_(fs::absolute(trees_dir)),
          runDir_(run_dir.empty()
                      ? sourcemaps::SourceMapStore::inferRunDir(treesDir_)
                      : fs::absolute(run_dir)),
          raw_(raw),
          colsPerShard_(cols_per_shard),
          treeIds_(std::move(tree_ids)),
          store_(runDir_, colsPerShard_),
          psi_(std::move(initial)),
          predictor_(treesDir_.string()) {
        if (treeIds_.empty()) treeIds_ = discover_tree_ids(treesDir_);
        if (treeIds_.empty()) throw std::runtime_error("No tree_*.bin found");
    }

    PsiStepSummary step(int threads) {
        ProbabilityRow before = psi_;
        psi_ = phi_native(predictor_, before, treeIds_, threads);
        ++step_;
        return summarize_step(before, psi_, treeIds_, step_);
    }

    std::vector<PsiStepSummary> iterate(
            int steps, int threads, double tol) {
        if (steps < 0) throw std::invalid_argument("steps must be >= 0");
        if (tol < 0.0) throw std::invalid_argument("tol must be >= 0");
        std::vector<PsiStepSummary> history;
        history.reserve(static_cast<size_t>(steps));
        for (int i = 0; i < steps; ++i) {
            PsiStepSummary s = step(threads);
            history.push_back(s);
            if (tol > 0.0 && s.max_tv <= tol) break;
        }
        return history;
    }


    const ProbabilityRow& psi() const { return psi_; }
    int stepCount() const { return step_; }
    const std::vector<int>& treeIds() const { return treeIds_; }
    sourcemaps::SourceMapStore& store() { return store_; }
    bool raw() const { return raw_; }

private:
    fs::path treesDir_;
    fs::path runDir_;
    bool raw_;
    int colsPerShard_;
    std::vector<int> treeIds_;
    sourcemaps::SourceMapStore store_;
    ProbabilityRow psi_;
    PredictDistribution predictor_;
    int step_ = 0;
};


static uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

template <class URNG>
static int sample_counts_code(const std::map<int, int>& counts, URNG& rng) {
    uint64_t total = 0;
    for (const auto& kv : counts) {
        if (kv.first != 0 && kv.second > 0) {
            total += static_cast<uint64_t>(kv.second);
        }
    }
    if (total == 0) {
        throw std::runtime_error("JointParticleState: predictor returned zero sampleable mass");
    }
    std::uniform_int_distribution<uint64_t> draw(1, total);
    const uint64_t u = draw(rng);
    uint64_t acc = 0;
    for (const auto& kv : counts) {
        if (kv.first == 0 || kv.second <= 0) continue;
        acc += static_cast<uint64_t>(kv.second);
        if (u <= acc) return kv.first;
    }
    throw std::logic_error("JointParticleState: categorical sampler fell through");
}

struct JointStepSummary {
    int step = 0;
    double mean_tv = 0.0;
    double max_tv = 0.0;
    int max_col = -1;
    int updates_per_particle = 0;
};

class JointParticleState {
public:
    JointParticleState(const std::string& trees_dir,
                       std::vector<int> base_row,
                       size_t n_particles,
                       uint64_t seed,
                       bool raw,
                       std::string run_dir,
                       int cols_per_shard,
                       std::vector<int> tree_ids)
        : treesDir_(fs::absolute(trees_dir)),
          runDir_(run_dir.empty()
                      ? sourcemaps::SourceMapStore::inferRunDir(treesDir_)
                      : fs::absolute(run_dir)),
          raw_(raw),
          colsPerShard_(cols_per_shard),
          treeIds_(std::move(tree_ids)),
          store_(runDir_, colsPerShard_),
          predictor_(treesDir_.string()),
          seed_(seed ? seed : 1ULL) {
        if (treeIds_.empty()) treeIds_ = discover_tree_ids(treesDir_);
        if (treeIds_.empty()) throw std::runtime_error("No tree_*.bin found");
        if (base_row.empty()) throw std::invalid_argument("base row must be non-empty");
        if (n_particles == 0) throw std::invalid_argument("n_particles must be > 0");
        particles_.assign(n_particles, base_row);
    }

    size_t particleCount() const { return particles_.size(); }
    size_t width() const { return particles_.empty() ? 0 : particles_[0].size(); }
    int stepCount() const { return step_; }

    void initializeMissing(int threads) {
        const uint64_t epoch = epoch_++;
        const int nt = threads > 0 ? threads : omp_get_max_threads();
        std::atomic<bool> failed{false};
        std::exception_ptr first_error;
        std::mutex error_mutex;

        #pragma omp parallel for schedule(dynamic) num_threads(nt)
        for (ssize_t pi = 0; pi < static_cast<ssize_t>(particles_.size()); ++pi) {
            if (failed.load(std::memory_order_relaxed)) continue;
            try {
                auto& row = particles_[static_cast<size_t>(pi)];
                std::vector<int> missing;
                missing.reserve(treeIds_.size());
                for (int tid : treeIds_) {
                    if (tid >= 0 && static_cast<size_t>(tid) < row.size() &&
                        row[static_cast<size_t>(tid)] == 0) {
                        missing.push_back(tid);
                    }
                }
                std::mt19937_64 rng(splitmix64(
                    seed_ ^ (epoch * 0x9e3779b97f4a7c15ULL) ^
                    static_cast<uint64_t>(pi)));
                std::shuffle(missing.begin(), missing.end(), rng);
                for (int tid : missing) {
                    const auto counts = predictor_.predict(tid, row);
                    row[static_cast<size_t>(tid)] = sample_counts_code(counts, rng);
                }
            } catch (...) {
                failed.store(true, std::memory_order_relaxed);
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!first_error) first_error = std::current_exception();
            }
        }
        if (first_error) std::rethrow_exception(first_error);
    }

    JointStepSummary step(int target_col, int threads) {
        const ProbabilityRow before = marginalsNative();
        const uint64_t epoch = epoch_++;
        const int nt = threads > 0 ? threads : omp_get_max_threads();
        std::atomic<bool> failed{false};
        std::exception_ptr first_error;
        std::mutex error_mutex;

        if (target_col >= 0 &&
            std::find(treeIds_.begin(), treeIds_.end(), target_col) == treeIds_.end()) {
            throw std::invalid_argument("target_col has no learned tree");
        }

        #pragma omp parallel for schedule(dynamic) num_threads(nt)
        for (ssize_t pi = 0; pi < static_cast<ssize_t>(particles_.size()); ++pi) {
            if (failed.load(std::memory_order_relaxed)) continue;
            try {
                auto& row = particles_[static_cast<size_t>(pi)];
                std::mt19937_64 rng(splitmix64(
                    seed_ ^ (epoch * 0x9e3779b97f4a7c15ULL) ^
                    static_cast<uint64_t>(pi)));
                int tid = target_col;
                if (tid < 0) {
                    std::uniform_int_distribution<size_t> pick(0, treeIds_.size() - 1);
                    tid = treeIds_[pick(rng)];
                }
                const auto counts = predictor_.predict(tid, row);
                row[static_cast<size_t>(tid)] = sample_counts_code(counts, rng);
            } catch (...) {
                failed.store(true, std::memory_order_relaxed);
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!first_error) first_error = std::current_exception();
            }
        }
        if (first_error) std::rethrow_exception(first_error);
        ++step_;
        return summarizeMarginalChange(before, marginalsNative(), 1);
    }

    JointStepSummary sweep(const std::string& mode,
                           int updates_per_particle,
                           int threads) {
        if (mode != "random_scan" && mode != "random_permutation") {
            throw std::invalid_argument(
                "mode must be 'random_scan' or 'random_permutation'");
        }
        int n_updates = updates_per_particle;
        if (n_updates <= 0) n_updates = static_cast<int>(treeIds_.size());

        const ProbabilityRow before = marginalsNative();
        const uint64_t epoch = epoch_++;
        const int nt = threads > 0 ? threads : omp_get_max_threads();
        std::atomic<bool> failed{false};
        std::exception_ptr first_error;
        std::mutex error_mutex;

        #pragma omp parallel for schedule(dynamic) num_threads(nt)
        for (ssize_t pi = 0; pi < static_cast<ssize_t>(particles_.size()); ++pi) {
            if (failed.load(std::memory_order_relaxed)) continue;
            try {
                auto& row = particles_[static_cast<size_t>(pi)];
                std::mt19937_64 rng(splitmix64(
                    seed_ ^ (epoch * 0x9e3779b97f4a7c15ULL) ^
                    static_cast<uint64_t>(pi)));

                if (mode == "random_permutation") {
                    std::vector<int> order = treeIds_;
                    std::shuffle(order.begin(), order.end(), rng);
                    const int use = std::min<int>(
                        n_updates, static_cast<int>(order.size()));
                    for (int k = 0; k < use; ++k) {
                        const int tid = order[static_cast<size_t>(k)];
                        const auto counts = predictor_.predict(tid, row);
                        row[static_cast<size_t>(tid)] =
                            sample_counts_code(counts, rng);
                    }
                } else {
                    std::uniform_int_distribution<size_t> pick(
                        0, treeIds_.size() - 1);
                    for (int k = 0; k < n_updates; ++k) {
                        const int tid = treeIds_[pick(rng)];
                        const auto counts = predictor_.predict(tid, row);
                        row[static_cast<size_t>(tid)] =
                            sample_counts_code(counts, rng);
                    }
                }
            } catch (...) {
                failed.store(true, std::memory_order_relaxed);
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!first_error) first_error = std::current_exception();
            }
        }
        if (first_error) std::rethrow_exception(first_error);
        ++step_;
        return summarizeMarginalChange(before, marginalsNative(), n_updates);
    }

    ProbabilityRow marginalsNative() const {
        ProbabilityRow out(width());
        const double inv = 1.0 / static_cast<double>(particles_.size());
        for (const auto& row : particles_) {
            for (size_t j = 0; j < row.size(); ++j) {
                const int code = row[j];
                if (code != 0) out[j][code] += inv;
            }
        }
        return out;
    }

    const std::vector<int>& particle(size_t i) const {
        if (i >= particles_.size()) throw std::out_of_range("particle index out of range");
        return particles_[i];
    }

    sourcemaps::SourceMapStore& store() { return store_; }
    bool raw() const { return raw_; }

private:
    JointStepSummary summarizeMarginalChange(
            const ProbabilityRow& before,
            const ProbabilityRow& after,
            int updates_per_particle) {
        JointStepSummary out;
        out.step = step_;
        out.updates_per_particle = updates_per_particle;
        double total = 0.0;
        int used = 0;
        for (int tid : treeIds_) {
            if (tid < 0 ||
                static_cast<size_t>(tid) >= before.size() ||
                static_cast<size_t>(tid) >= after.size()) continue;
            const double d = tv_distance(
                before[static_cast<size_t>(tid)],
                after[static_cast<size_t>(tid)]);
            total += d;
            ++used;
            if (d > out.max_tv) {
                out.max_tv = d;
                out.max_col = tid;
            }
        }
        out.mean_tv = used ? total / static_cast<double>(used) : 0.0;
        return out;
    }

    fs::path treesDir_;
    fs::path runDir_;
    bool raw_;
    int colsPerShard_;
    std::vector<int> treeIds_;
    sourcemaps::SourceMapStore store_;
    PredictDistribution predictor_;
    uint64_t seed_;
    uint64_t epoch_ = 1;
    int step_ = 0;
    std::vector<std::vector<int>> particles_;
};

static py::dict joint_summary_to_py(const JointStepSummary& s) {
    py::dict out;
    out["step"] = s.step;
    out["mean_tv"] = s.mean_tv;
    out["max_tv"] = s.max_tv;
    out["max_col"] = s.max_col;
    out["updates_per_particle"] = s.updates_per_particle;
    return out;
}

static std::unique_ptr<JointParticleState> joint_state_from_row_py(
        const std::string& trees_dir,
        py::array row,
        size_t n_particles,
        uint64_t seed,
        bool raw,
        std::string run_dir,
        int cols_per_shard,
        std::vector<int> tree_ids,
        bool initialize_missing,
        int threads) {
    fs::path treesDir = fs::absolute(trees_dir);
    fs::path runDir = run_dir.empty()
                          ? sourcemaps::SourceMapStore::inferRunDir(treesDir)
                          : fs::absolute(run_dir);
    if (raw && (runDir.empty() || !fs::exists(runDir / "source_maps"))) {
        throw std::runtime_error("When raw=True, run_dir must contain source_maps/");
    }
    sourcemaps::SourceMapStore store(runDir, cols_per_shard);
    std::vector<int> codes = encode_row_codes(store, row, raw);
    auto state = std::make_unique<JointParticleState>(
        trees_dir, std::move(codes), n_particles, seed, raw, run_dir,
        cols_per_shard, std::move(tree_ids));
    if (initialize_missing) {
        py::gil_scoped_release release;
        state->initializeMissing(threads);
    }
    return state;
}


struct LdpStepSummary {
    int step = 0;
    int target_col = -1;
    int n = 0;
    std::string event_mode;
    double event_tv = 0.0;
    double sanov_rate = 0.0;
    double sanov_exponent = 0.0;
    double mean_tv = 0.0;
    double max_tv = 0.0;
    int max_col = -1;
    int projected_count = 0;
};

class LdpPsiState {
public:
    LdpPsiState(const std::string& trees_dir,
                ProbabilityRow initial,
                bool raw,
                std::string run_dir,
                int cols_per_shard,
                std::vector<int> tree_ids)
        : treesDir_(fs::absolute(trees_dir)),
          runDir_(run_dir.empty()
                      ? sourcemaps::SourceMapStore::inferRunDir(treesDir_)
                      : fs::absolute(run_dir)),
          raw_(raw),
          colsPerShard_(cols_per_shard),
          treeIds_(std::move(tree_ids)),
          store_(runDir_, colsPerShard_),
          psi_(std::move(initial)),
          predictor_(treesDir_.string()) {
        if (treeIds_.empty()) treeIds_ = discover_tree_ids(treesDir_);
        if (treeIds_.empty()) throw std::runtime_error("No tree_*.bin found");
        predictor_.preload(treeIds_);
    }

    LdpStepSummary step(int target_col,
                        int n,
                        const std::string& event,
                        uint64_t seed,
                        double response_scale,
                        int threads) {
        if (!(response_scale >= 0.0) || !std::isfinite(response_scale)) {
            throw std::invalid_argument(
                "response_scale must be finite and non-negative");
        }

        const ProbabilityRow before = psi_;
        uint64_t event_seed = splitmix64(
            (seed ? seed : 1ULL) ^
            static_cast<uint64_t>(step_ + 1));

        if (target_col < 0) {
            std::mt19937_64 rng(event_seed);
            std::uniform_int_distribution<size_t> pick(
                0, treeIds_.size() - 1);
            target_col = treeIds_[pick(rng)];
        }
        if (target_col < 0 ||
            static_cast<size_t>(target_col) >= psi_.size()) {
            throw std::out_of_range("target_col out of range");
        }
        if (std::find(treeIds_.begin(), treeIds_.end(), target_col) ==
            treeIds_.end()) {
            throw std::invalid_argument("target_col has no learned tree");
        }

        ProbabilityDistribution empirical;
        std::string event_mode;
        if (event == "zero_action" || n == 0) {
            empirical = psi_[static_cast<size_t>(target_col)];
            event_mode = "zero_action";
        } else if (event == "mode") {
            empirical = multinomial_mode_empirical(
                psi_[static_cast<size_t>(target_col)], n);
            event_mode = "mode";
        } else if (event == "sample") {
            empirical = multinomial_empirical(
                psi_[static_cast<size_t>(target_col)], n, event_seed);
            event_mode = "sample";
        } else {
            throw std::invalid_argument(
                "event must be 'zero_action', 'mode', or 'sample'");
        }

        ProbabilityRow perturbed = psi_;
        perturbed[static_cast<size_t>(target_col)] = empirical;

        const ProbabilityRow base_response =
            phi_native(predictor_, psi_, treeIds_, threads);
        const ProbabilityRow pert_response =
            phi_native(predictor_, perturbed, treeIds_, threads);

        ProbabilityRow next = psi_;
        next[static_cast<size_t>(target_col)] = empirical;

        int projected_count = 0;
        for (int tid : treeIds_) {
            if (tid == target_col || tid < 0 ||
                static_cast<size_t>(tid) >= psi_.size()) continue;
            bool projected = false;
            next[static_cast<size_t>(tid)] = add_response_delta(
                psi_[static_cast<size_t>(tid)],
                base_response[static_cast<size_t>(tid)],
                pert_response[static_cast<size_t>(tid)],
                response_scale,
                &projected);
            if (projected) ++projected_count;
        }

        ++step_;
        const PsiStepSummary move =
            summarize_step(before, next, treeIds_, step_);

        LdpStepSummary out;
        out.step = step_;
        out.target_col = target_col;
        out.n = (event_mode == "mode" || event_mode == "sample") ? n : 0;
        out.event_mode = event_mode;
        out.event_tv = tv_distance(
            before[static_cast<size_t>(target_col)], empirical);
        out.sanov_rate = kl_divergence(
            empirical, before[static_cast<size_t>(target_col)]);
        out.sanov_exponent =
            (out.n > 0 && std::isfinite(out.sanov_rate))
                ? static_cast<double>(out.n) * out.sanov_rate
                : 0.0;
        out.mean_tv = move.mean_tv;
        out.max_tv = move.max_tv;
        out.max_col = move.max_col;
        out.projected_count = projected_count;

        psi_ = std::move(next);
        return out;
    }


    LdpStepSummary sweep(
            int n,
            const std::string& event,
            uint64_t seed,
            double response_scale,
            int threads,
            bool random_permutation) {
        const ProbabilityRow before = psi_;

        std::vector<int> order = treeIds_;
        if (random_permutation) {
            std::mt19937_64 rng(seed ? seed : 1ULL);
            std::shuffle(order.begin(), order.end(), rng);
        }

        double total_event_tv = 0.0;
        double total_rate = 0.0;
        int projected_count = 0;

        for (size_t k = 0; k < order.size(); ++k) {
            const uint64_t local_seed = splitmix64(
                (seed ? seed : 1ULL) ^
                static_cast<uint64_t>(k + 1) ^
                (static_cast<uint64_t>(step_ + 1) << 32));
            const LdpStepSummary h = step(
                order[k], n, event, local_seed,
                response_scale, threads);
            total_event_tv += h.event_tv;
            if (std::isfinite(h.sanov_rate)) total_rate += h.sanov_rate;
            projected_count += h.projected_count;
        }

        const PsiStepSummary move =
            summarize_step(before, psi_, treeIds_, step_);

        LdpStepSummary out;
        out.step = step_;
        out.target_col = -1;
        out.n = (event == "mode" || event == "sample") ? n : 0;
        out.event_mode = std::string("sweep_") + event;
        out.event_tv = order.empty()
            ? 0.0
            : total_event_tv / static_cast<double>(order.size());
        out.sanov_rate = total_rate;
        out.sanov_exponent =
            (out.n > 0 && std::isfinite(total_rate))
                ? static_cast<double>(out.n) * total_rate
                : 0.0;
        out.mean_tv = move.mean_tv;
        out.max_tv = move.max_tv;
        out.max_col = move.max_col;
        out.projected_count = projected_count;
        return out;
    }

    std::vector<LdpStepSummary> iterate(
            int steps,
            int n,
            const std::string& event,
            uint64_t seed,
            double response_scale,
            int threads,
            double tol,
            int patience) {
        if (steps < 0) throw std::invalid_argument("steps must be >= 0");
        if (tol < 0.0) throw std::invalid_argument("tol must be >= 0");
        if (patience < 1) throw std::invalid_argument("patience must be >= 1");

        std::vector<LdpStepSummary> history;
        history.reserve(static_cast<size_t>(steps));
        int quiet = 0;
        for (int k = 0; k < steps; ++k) {
            LdpStepSummary h = step(
                -1, n, event,
                splitmix64((seed ? seed : 1ULL) ^ static_cast<uint64_t>(k + 1)),
                response_scale, threads);
            history.push_back(h);
            if (tol > 0.0 && h.max_tv <= tol) {
                ++quiet;
                if (quiet >= patience) break;
            } else {
                quiet = 0;
            }
        }
        return history;
    }

    std::vector<LdpStepSummary> iterateSweeps(
            int sweeps,
            int n,
            const std::string& event,
            uint64_t seed,
            double response_scale,
            int threads,
            double tol,
            int patience,
            bool random_permutation) {
        if (sweeps < 0) throw std::invalid_argument("sweeps must be >= 0");
        if (tol < 0.0) throw std::invalid_argument("tol must be >= 0");
        if (patience < 1) throw std::invalid_argument("patience must be >= 1");

        std::vector<LdpStepSummary> history;
        history.reserve(static_cast<size_t>(sweeps));
        int quiet = 0;
        for (int s = 0; s < sweeps; ++s) {
            const uint64_t sweep_seed = splitmix64(
                (seed ? seed : 1ULL) ^
                static_cast<uint64_t>(s + 1));
            LdpStepSummary h = sweep(
                n, event, sweep_seed, response_scale,
                threads, random_permutation);
            history.push_back(h);

            if (tol > 0.0 && h.max_tv <= tol) {
                ++quiet;
                if (quiet >= patience) break;
            } else {
                quiet = 0;
            }
        }
        return history;
    }

    const ProbabilityRow& psi() const { return psi_; }
    int stepCount() const { return step_; }
    size_t size() const { return psi_.size(); }
    sourcemaps::SourceMapStore& store() { return store_; }
    bool raw() const { return raw_; }

private:
    fs::path treesDir_;
    fs::path runDir_;
    bool raw_;
    int colsPerShard_;
    std::vector<int> treeIds_;
    sourcemaps::SourceMapStore store_;
    ProbabilityRow psi_;
    PredictDistribution predictor_;
    int step_ = 0;
};

static py::dict ldp_step_summary_to_py(const LdpStepSummary& s) {
    py::dict out;
    out["step"] = s.step;
    out["target_col"] = s.target_col;
    out["event_mode"] = s.event_mode;
    out["n"] = s.n > 0 ? py::cast(s.n) : py::none();
    out["event_tv"] = s.event_tv;
    out["sanov_rate"] = s.sanov_rate;
    out["sanov_exponent"] =
        s.n > 0 ? py::cast(s.sanov_exponent) : py::none();
    out["mean_tv"] = s.mean_tv;
    out["max_tv"] = s.max_tv;
    out["max_col"] = s.max_col;
    out["projected_count"] = s.projected_count;
    return out;
}


static std::unique_ptr<LdpPsiState> ldp_state_from_psi_py(
        const std::string& trees_dir,
        py::object psi_obj,
        bool raw,
        std::string run_dir,
        int cols_per_shard,
        std::vector<int> tree_ids) {
    fs::path treesDir = fs::absolute(trees_dir);
    fs::path runDir = run_dir.empty()
                          ? sourcemaps::SourceMapStore::inferRunDir(treesDir)
                          : fs::absolute(run_dir);
    if (raw && (runDir.empty() || !fs::exists(runDir / "source_maps"))) {
        throw std::runtime_error(
            "When raw=True, run_dir must contain source_maps/");
    }
    if (tree_ids.empty()) tree_ids = discover_tree_ids(treesDir);
    if (tree_ids.empty()) throw std::runtime_error("No tree_*.bin found");

    sourcemaps::SourceMapStore store(runDir, cols_per_shard);
    ProbabilityRow psi = python_to_psi(store, psi_obj, raw);

    return std::make_unique<LdpPsiState>(
        trees_dir, std::move(psi), raw, run_dir,
        cols_per_shard, std::move(tree_ids));
}

static std::unique_ptr<LdpPsiState> ldp_state_from_row_py(
        const std::string& trees_dir,
        py::array row,
        bool raw,
        std::string run_dir,
        int cols_per_shard,
        std::vector<int> tree_ids,
        int threads) {
    fs::path treesDir = fs::absolute(trees_dir);
    fs::path runDir = run_dir.empty()
                          ? sourcemaps::SourceMapStore::inferRunDir(treesDir)
                          : fs::absolute(run_dir);
    if (raw && (runDir.empty() || !fs::exists(runDir / "source_maps"))) {
        throw std::runtime_error(
            "When raw=True, run_dir must contain source_maps/");
    }
    if (tree_ids.empty()) tree_ids = discover_tree_ids(treesDir);
    if (tree_ids.empty()) throw std::runtime_error("No tree_*.bin found");

    sourcemaps::SourceMapStore store(runDir, cols_per_shard);
    const std::vector<int> row_codes = encode_row_codes(store, row, raw);
    PredictDistribution predictor(treesDir.string());
    predictor.preload(tree_ids);

    ProbabilityRow psi(row_codes.size());
    std::unordered_set<int> learned(tree_ids.begin(), tree_ids.end());

    const int nt = threads > 0 ? threads : omp_get_max_threads();
    std::atomic<bool> failed{false};
    std::exception_ptr first_error;
    std::mutex error_mutex;

    {
        py::gil_scoped_release release;
        #pragma omp parallel for schedule(dynamic) num_threads(nt)
        for (ssize_t k = 0; k < static_cast<ssize_t>(tree_ids.size()); ++k) {
            if (failed.load(std::memory_order_relaxed)) continue;
            try {
                const int tid = tree_ids[static_cast<size_t>(k)];
                if (tid < 0 ||
                    static_cast<size_t>(tid) >= row_codes.size()) continue;
                std::vector<int> conditioned = row_codes;
                conditioned[static_cast<size_t>(tid)] = 0;
                psi[static_cast<size_t>(tid)] =
                    predictor.predictProbability(tid, conditioned);
            } catch (...) {
                failed.store(true, std::memory_order_relaxed);
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!first_error) first_error = std::current_exception();
            }
        }
    }
    if (first_error) std::rethrow_exception(first_error);

    for (size_t i = 0; i < psi.size(); ++i) {
        if (!psi[i].empty()) continue;
        const int code = row_codes[i];
        if (code != 0) {
            psi[i][code] = 1.0;
        } else if (!learned.count(static_cast<int>(i))) {
            throw std::runtime_error(
                "Cannot initialize missing non-target column " +
                std::to_string(i) + " from a hard row");
        }
    }

    return std::make_unique<LdpPsiState>(
        trees_dir, std::move(psi), raw, run_dir,
        cols_per_shard, std::move(tree_ids));
}



/*
 * Centered hard-response LDP dynamics.
 *
 * For a probability-valued state Psi=(p_1,...,p_d), choose source i and an
 * empirical realization nu_i of p_i.  Let delta_i = nu_i - p_i.  For every
 * dependent target j != i define the fixed hard-response kernel
 *
 *   K_{j<-i}(.|sigma) = phi_j(x_empty with X_i=sigma).
 *
 * The update is
 *
 *   p_i' = nu_i,
 *   p_j' = p_j + response_scale * sum_sigma delta_i(sigma) K_{j<-i}(.|sigma).
 *
 * Targets whose learned tree never uses i are unchanged.  If delta_i=0 the
 * whole state is unchanged exactly, without assuming the tower identity
 * sum_sigma p_i(sigma) K_{j<-i}(.|sigma)=p_j.
 */

struct CenteredKernelKey {
    int source = -1;
    int target = -1;
    int sigma = 0;

    bool operator==(const CenteredKernelKey& other) const {
        return source == other.source &&
               target == other.target &&
               sigma == other.sigma;
    }
};

struct CenteredKernelKeyHash {
    size_t operator()(const CenteredKernelKey& k) const noexcept {
        size_t h = static_cast<size_t>(k.source + 0x9e3779b9);
        h ^= static_cast<size_t>(k.target + 0x85ebca6b) +
             (h << 6) + (h >> 2);
        h ^= static_cast<size_t>(k.sigma + 0xc2b2ae35) +
             (h << 6) + (h >> 2);
        return h;
    }
};

class CenteredHardKernelCache {
public:
    ProbabilityDistribution get(
            PredictDistribution& predictor,
            int source,
            int target,
            int sigma,
            size_t width) {
        const CenteredKernelKey key{source, target, sigma};

        {
            std::shared_lock<std::shared_mutex> lock(mutex_);
            auto it = cache_.find(key);
            if (it != cache_.end()) return it->second;
        }

        std::vector<int> row(width, 0);
        row[static_cast<size_t>(source)] = sigma;
        ProbabilityDistribution computed =
            predictor.predictProbability(target, row);

        {
            std::unique_lock<std::shared_mutex> lock(mutex_);
            auto inserted = cache_.emplace(key, computed);
            if (!inserted.second) return inserted.first->second;
        }
        return computed;
    }

    size_t size() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return cache_.size();
    }

private:
    mutable std::shared_mutex mutex_;
    std::unordered_map<
        CenteredKernelKey,
        ProbabilityDistribution,
        CenteredKernelKeyHash> cache_;
};

static ProbabilityDistribution centered_hard_response_delta(
        PredictDistribution& predictor,
        CenteredHardKernelCache& kernel_cache,
        int source,
        int target,
        const ProbabilityDistribution& delta,
        size_t width) {
    ProbabilityDistribution response;

    for (const auto& kv : delta) {
        const int sigma = kv.first;
        const double weight = kv.second;
        if (std::abs(weight) <= 1e-18) continue;

        const ProbabilityDistribution q =
            kernel_cache.get(
                predictor, source, target, sigma, width);
        for (const auto& out : q) {
            response[out.first] += weight * out.second;
        }
    }
    return response;
}

static ProbabilityDistribution apply_centered_response(
        const ProbabilityDistribution& current,
        const ProbabilityDistribution& response,
        double response_scale,
        bool* projected) {
    std::set<int> keys;
    for (const auto& kv : current) keys.insert(kv.first);
    for (const auto& kv : response) keys.insert(kv.first);

    ProbabilityDistribution candidate;
    for (int code : keys) {
        const double cur =
            current.count(code) ? current.at(code) : 0.0;
        const double dr =
            response.count(code) ? response.at(code) : 0.0;
        candidate[code] = cur + response_scale * dr;
    }
    return simplex_project(candidate, projected);
}

static ProbabilityDistribution distribution_difference(
        const ProbabilityDistribution& a,
        const ProbabilityDistribution& b) {
    std::set<int> keys;
    for (const auto& kv : a) keys.insert(kv.first);
    for (const auto& kv : b) keys.insert(kv.first);

    ProbabilityDistribution out;
    for (int code : keys) {
        const double av = a.count(code) ? a.at(code) : 0.0;
        const double bv = b.count(code) ? b.at(code) : 0.0;
        const double d = av - bv;
        if (std::abs(d) > 1e-18) out[code] = d;
    }
    return out;
}

class CenteredLdpPsiState {
public:
    CenteredLdpPsiState(
            const std::string& trees_dir,
            ProbabilityRow initial,
            bool raw,
            std::string run_dir,
            int cols_per_shard,
            std::vector<int> tree_ids)
        : treesDir_(fs::absolute(trees_dir)),
          runDir_(run_dir.empty()
                      ? sourcemaps::SourceMapStore::inferRunDir(treesDir_)
                      : fs::absolute(run_dir)),
          raw_(raw),
          colsPerShard_(cols_per_shard),
          treeIds_(std::move(tree_ids)),
          store_(runDir_, colsPerShard_),
          psi_(std::move(initial)),
          predictor_(std::make_shared<PredictDistribution>(treesDir_.string())),
          kernelCache_(std::make_shared<CenteredHardKernelCache>()) {
        if (treeIds_.empty()) treeIds_ = discover_tree_ids(treesDir_);
        if (treeIds_.empty()) throw std::runtime_error("No tree_*.bin found");
        predictor_->preload(treeIds_);

        const size_t width = psi_.size();
        dependencyTargets_.assign(width, {});
        std::unordered_set<int> learned(
            treeIds_.begin(), treeIds_.end());

        for (int target : treeIds_) {
            const std::set<int> used = predictor_->usedColumns(target);
            for (int source : used) {
                if (source == target ||
                    source < 0 ||
                    static_cast<size_t>(source) >= width ||
                    !learned.count(source)) {
                    continue;
                }
                dependencyTargets_[static_cast<size_t>(source)]
                    .push_back(target);
            }
        }

        for (auto& targets : dependencyTargets_) {
            std::sort(targets.begin(), targets.end());
            targets.erase(
                std::unique(targets.begin(), targets.end()),
                targets.end());
        }
    }

    CenteredLdpPsiState(
            const std::string& trees_dir,
            ProbabilityRow initial,
            bool raw,
            std::string run_dir,
            int cols_per_shard,
            std::vector<int> tree_ids,
            std::shared_ptr<PredictDistribution> shared_predictor,
            std::shared_ptr<CenteredHardKernelCache> shared_cache,
            std::shared_ptr<const std::vector<std::vector<int>>> shared_dependencies)
        : treesDir_(fs::absolute(trees_dir)),
          runDir_(run_dir.empty()
                      ? sourcemaps::SourceMapStore::inferRunDir(treesDir_)
                      : fs::absolute(run_dir)),
          raw_(raw),
          colsPerShard_(cols_per_shard),
          treeIds_(std::move(tree_ids)),
          store_(runDir_, colsPerShard_),
          psi_(std::move(initial)),
          predictor_(std::move(shared_predictor)),
          kernelCache_(std::move(shared_cache)),
          sharedDependencyTargets_(std::move(shared_dependencies)) {
        if (!predictor_) {
            throw std::invalid_argument("shared predictor is null");
        }
        if (!kernelCache_) {
            throw std::invalid_argument("shared kernel cache is null");
        }
        if (!sharedDependencyTargets_) {
            throw std::invalid_argument("shared dependency graph is null");
        }
    }

    LdpStepSummary applyEmpirical(
            int source,
            const ProbabilityDistribution& empirical,
            const std::string& event_mode,
            int n,
            double response_scale,
            int threads) {
        if (!(response_scale >= 0.0) ||
            !std::isfinite(response_scale)) {
            throw std::invalid_argument(
                "response_scale must be finite and non-negative");
        }
        if (source < 0 ||
            static_cast<size_t>(source) >= psi_.size()) {
            throw std::out_of_range("source column out of range");
        }
        if (std::find(
                treeIds_.begin(), treeIds_.end(), source) ==
            treeIds_.end()) {
            throw std::invalid_argument(
                "source column has no learned tree");
        }

        const ProbabilityRow before = psi_;

        const ProbabilityDistribution delta =
            distribution_difference(
                empirical,
                psi_[static_cast<size_t>(source)]);

        ProbabilityRow next = psi_;
        next[static_cast<size_t>(source)] = empirical;

        const auto& targets =
            sharedDependencyTargets_
                ? (*sharedDependencyTargets_)[static_cast<size_t>(source)]
                : dependencyTargets_[static_cast<size_t>(source)];

        const int nt =
            threads > 0 ? threads : omp_get_max_threads();
        std::atomic<int> projected_count{0};
        std::atomic<bool> failed{false};
        std::exception_ptr first_error;
        std::mutex error_mutex;

        #pragma omp parallel for schedule(dynamic) num_threads(nt)
        for (ssize_t k = 0;
             k < static_cast<ssize_t>(targets.size());
             ++k) {
            if (failed.load(std::memory_order_relaxed)) continue;
            try {
                const int target =
                    targets[static_cast<size_t>(k)];

                if (clamped_.count(target)) {
                    continue;
                }

                const ProbabilityDistribution response =
                    centered_hard_response_delta(
                        *predictor_,
                        *kernelCache_,
                        source,
                        target,
                        delta,
                        psi_.size());

                bool projected = false;
                next[static_cast<size_t>(target)] =
                    apply_centered_response(
                        psi_[static_cast<size_t>(target)],
                        response,
                        response_scale,
                        &projected);

                if (projected) {
                    projected_count.fetch_add(
                        1, std::memory_order_relaxed);
                }
            } catch (...) {
                failed.store(true, std::memory_order_relaxed);
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!first_error) {
                    first_error = std::current_exception();
                }
            }
        }
        if (first_error) std::rethrow_exception(first_error);

        ++step_;
        const PsiStepSummary move =
            summarize_step(before, next, treeIds_, step_);

        LdpStepSummary out;
        out.step = step_;
        out.target_col = source;
        out.n = n;
        out.event_mode = event_mode;
        out.event_tv = tv_distance(
            before[static_cast<size_t>(source)],
            empirical);
        out.sanov_rate = kl_divergence(
            empirical,
            before[static_cast<size_t>(source)]);
        out.sanov_exponent =
            (n > 0 && std::isfinite(out.sanov_rate))
                ? static_cast<double>(n) * out.sanov_rate
                : 0.0;
        out.mean_tv = move.mean_tv;
        out.max_tv = move.max_tv;
        out.max_col = move.max_col;
        out.projected_count =
            projected_count.load(std::memory_order_relaxed);

        psi_ = std::move(next);
        return out;
    }

    LdpStepSummary hardObserve(
            int source,
            int sigma,
            double response_scale,
            int threads,
            bool clamp) {
        if (sigma == 0) {
            throw std::invalid_argument(
                "hard observation cannot use missing code 0");
        }
        ProbabilityDistribution empirical;
        empirical[sigma] = 1.0;

        // Apply the observation against the pre-observation state first,
        // then register the coordinate as persistent evidence.
        LdpStepSummary out = applyEmpirical(
            source,
            empirical,
            "hard_observation",
            0,
            response_scale,
            threads);

        if (clamp) {
            clamped_[source] = sigma;
        }
        return out;
    }

    LdpStepSummary step(
            int source,
            int n,
            const std::string& event,
            uint64_t seed,
            double response_scale,
            int threads) {
        if (!(response_scale >= 0.0) ||
            !std::isfinite(response_scale)) {
            throw std::invalid_argument(
                "response_scale must be finite and non-negative");
        }

        const ProbabilityRow before = psi_;
        const uint64_t event_seed = splitmix64(
            (seed ? seed : 1ULL) ^
            static_cast<uint64_t>(step_ + 1));

        if (source < 0) {
            std::mt19937_64 rng(event_seed);
            std::uniform_int_distribution<size_t> pick(
                0, treeIds_.size() - 1);
            source = treeIds_[pick(rng)];
        }
        if (source < 0 ||
            static_cast<size_t>(source) >= psi_.size()) {
            throw std::out_of_range("source column out of range");
        }
        if (std::find(
                treeIds_.begin(), treeIds_.end(), source) ==
            treeIds_.end()) {
            throw std::invalid_argument(
                "source column has no learned tree");
        }

        auto clamp_it = clamped_.find(source);
        if (clamp_it != clamped_.end()) {
            ProbabilityDistribution empirical;
            empirical[clamp_it->second] = 1.0;
            return applyEmpirical(
                source,
                empirical,
                "clamped",
                0,
                response_scale,
                threads);
        }

        ProbabilityDistribution empirical;
        std::string event_mode;
        if (event == "zero_action" || n == 0) {
            empirical = psi_[static_cast<size_t>(source)];
            event_mode = "zero_action";
        } else if (event == "mode") {
            empirical = multinomial_mode_empirical(
                psi_[static_cast<size_t>(source)], n);
            event_mode = "mode";
        } else if (event == "sample") {
            empirical = multinomial_empirical(
                psi_[static_cast<size_t>(source)],
                n,
                event_seed);
            event_mode = "sample";
        } else {
            throw std::invalid_argument(
                "event must be 'zero_action', 'mode', or 'sample'");
        }

        return applyEmpirical(
            source,
            empirical,
            event_mode,
            (event_mode == "mode" || event_mode == "sample") ? n : 0,
            response_scale,
            threads);
    }

    LdpStepSummary sweep(
            int n,
            const std::string& event,
            uint64_t seed,
            double response_scale,
            int threads,
            bool random_permutation) {
        const ProbabilityRow before = psi_;

        std::vector<int> order = treeIds_;
        if (random_permutation) {
            std::mt19937_64 rng(seed ? seed : 1ULL);
            std::shuffle(order.begin(), order.end(), rng);
        }

        double total_event_tv = 0.0;
        double total_rate = 0.0;
        int projected_count = 0;

        for (size_t k = 0; k < order.size(); ++k) {
            const uint64_t local_seed = splitmix64(
                (seed ? seed : 1ULL) ^
                static_cast<uint64_t>(k + 1) ^
                (static_cast<uint64_t>(step_ + 1) << 32));

            const LdpStepSummary h = step(
                order[k],
                n,
                event,
                local_seed,
                response_scale,
                threads);

            total_event_tv += h.event_tv;
            if (std::isfinite(h.sanov_rate)) {
                total_rate += h.sanov_rate;
            }
            projected_count += h.projected_count;
        }

        const PsiStepSummary move =
            summarize_step(before, psi_, treeIds_, step_);

        LdpStepSummary out;
        out.step = step_;
        out.target_col = -1;
        out.n =
            (event == "mode" || event == "sample")
                ? n
                : 0;
        out.event_mode =
            std::string("centered_sweep_") + event;
        out.event_tv =
            order.empty()
                ? 0.0
                : total_event_tv /
                    static_cast<double>(order.size());
        out.sanov_rate = total_rate;
        out.sanov_exponent =
            (out.n > 0 && std::isfinite(total_rate))
                ? static_cast<double>(out.n) * total_rate
                : 0.0;
        out.mean_tv = move.mean_tv;
        out.max_tv = move.max_tv;
        out.max_col = move.max_col;
        out.projected_count = projected_count;
        return out;
    }

    std::vector<LdpStepSummary> iterateSweeps(
            int sweeps,
            int n,
            const std::string& event,
            uint64_t seed,
            double response_scale,
            int threads,
            double tol,
            int patience,
            bool random_permutation) {
        if (sweeps < 0) {
            throw std::invalid_argument(
                "sweeps must be >= 0");
        }
        if (tol < 0.0) {
            throw std::invalid_argument(
                "tol must be >= 0");
        }
        if (patience < 1) {
            throw std::invalid_argument(
                "patience must be >= 1");
        }

        std::vector<LdpStepSummary> history;
        history.reserve(static_cast<size_t>(sweeps));
        int quiet = 0;

        for (int s = 0; s < sweeps; ++s) {
            const uint64_t sweep_seed = splitmix64(
                (seed ? seed : 1ULL) ^
                static_cast<uint64_t>(s + 1));

            LdpStepSummary h = sweep(
                n,
                event,
                sweep_seed,
                response_scale,
                threads,
                random_permutation);

            history.push_back(h);

            if (tol > 0.0 && h.max_tv <= tol) {
                ++quiet;
                if (quiet >= patience) break;
            } else {
                quiet = 0;
            }
        }
        return history;
    }


    void resetFromHardCodes(
            const std::vector<int>& row_codes,
            int threads) {
        if (row_codes.size() != psi_.size()) {
            throw std::invalid_argument(
                "row width does not match centered LDP state width");
        }

        ProbabilityRow next(row_codes.size());
        const int nt =
            threads > 0 ? threads : omp_get_max_threads();

        std::atomic<bool> failed{false};
        std::exception_ptr first_error;
        std::mutex error_mutex;

        #pragma omp parallel for schedule(dynamic) num_threads(nt)
        for (ssize_t k = 0;
             k < static_cast<ssize_t>(treeIds_.size());
             ++k) {
            if (failed.load(std::memory_order_relaxed)) continue;
            try {
                const int tid =
                    treeIds_[static_cast<size_t>(k)];
                if (tid < 0 ||
                    static_cast<size_t>(tid) >= row_codes.size()) {
                    continue;
                }

                std::vector<int> conditioned = row_codes;
                conditioned[static_cast<size_t>(tid)] = 0;

                next[static_cast<size_t>(tid)] =
                    predictor_->predictProbability(
                        tid, conditioned);
            } catch (...) {
                failed.store(true, std::memory_order_relaxed);
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!first_error) {
                    first_error = std::current_exception();
                }
            }
        }

        if (first_error) std::rethrow_exception(first_error);

        std::unordered_set<int> learned(
            treeIds_.begin(), treeIds_.end());

        for (size_t i = 0; i < next.size(); ++i) {
            if (!next[i].empty()) continue;
            const int code = row_codes[i];
            if (code != 0) {
                next[i][code] = 1.0;
            } else if (!learned.count(static_cast<int>(i))) {
                throw std::runtime_error(
                    "Cannot initialize missing non-target column " +
                    std::to_string(i) +
                    " from a hard row");
            }
        }

        psi_ = std::move(next);
        clamped_.clear();
        step_ = 0;
    }

    std::vector<int> hardCodes() const {
        std::vector<int> out(psi_.size(), 0);

        for (size_t i = 0; i < psi_.size(); ++i) {
            const auto& q = psi_[i];
            if (q.empty()) continue;

            bool have = false;
            int best_code = 0;
            double best_p =
                -std::numeric_limits<double>::infinity();

            for (const auto& kv : q) {
                if (!have || kv.second > best_p) {
                    have = true;
                    best_code = kv.first;
                    best_p = kv.second;
                }
            }
            out[i] = best_code;
        }
        return out;
    }

    size_t kernelCacheSize() const {
        return kernelCache_->size();
    }

    size_t clampedCount() const {
        return clamped_.size();
    }

    const ProbabilityRow& psi() const { return psi_; }
    int stepCount() const { return step_; }
    size_t size() const { return psi_.size(); }
    size_t dependencyCount(int source) const {
        if (source < 0 ||
            static_cast<size_t>(source) >=
                dependencyTargets_.size()) {
            throw std::out_of_range(
                "source column out of range");
        }
        if (sharedDependencyTargets_) {
            return (*sharedDependencyTargets_)[static_cast<size_t>(source)].size();
        }
        return dependencyTargets_[static_cast<size_t>(source)].size();
    }
    sourcemaps::SourceMapStore& store() { return store_; }
    bool raw() const { return raw_; }

private:
    fs::path treesDir_;
    fs::path runDir_;
    bool raw_;
    int colsPerShard_;
    std::vector<int> treeIds_;
    std::vector<std::vector<int>> dependencyTargets_;
    sourcemaps::SourceMapStore store_;
    ProbabilityRow psi_;
    std::shared_ptr<PredictDistribution> predictor_;
    std::shared_ptr<CenteredHardKernelCache> kernelCache_;
    std::shared_ptr<const std::vector<std::vector<int>>> sharedDependencyTargets_;
    std::unordered_map<int, int> clamped_;
    int step_ = 0;
};

static std::unique_ptr<CenteredLdpPsiState>
centered_ldp_state_from_psi_py(
        const std::string& trees_dir,
        py::object psi_obj,
        bool raw,
        std::string run_dir,
        int cols_per_shard,
        std::vector<int> tree_ids) {
    fs::path treesDir = fs::absolute(trees_dir);
    fs::path runDir =
        run_dir.empty()
            ? sourcemaps::SourceMapStore::inferRunDir(treesDir)
            : fs::absolute(run_dir);

    if (raw &&
        (runDir.empty() ||
         !fs::exists(runDir / "source_maps"))) {
        throw std::runtime_error(
            "When raw=True, run_dir must contain source_maps/");
    }

    if (tree_ids.empty()) {
        tree_ids = discover_tree_ids(treesDir);
    }
    if (tree_ids.empty()) {
        throw std::runtime_error("No tree_*.bin found");
    }

    sourcemaps::SourceMapStore store(
        runDir, cols_per_shard);
    ProbabilityRow psi =
        python_to_psi(store, psi_obj, raw);

    return std::make_unique<CenteredLdpPsiState>(
        trees_dir,
        std::move(psi),
        raw,
        run_dir,
        cols_per_shard,
        std::move(tree_ids));
}

static std::unique_ptr<CenteredLdpPsiState>
centered_ldp_state_from_row_py(
        const std::string& trees_dir,
        py::array row,
        bool raw,
        std::string run_dir,
        int cols_per_shard,
        std::vector<int> tree_ids,
        int threads) {
    fs::path treesDir = fs::absolute(trees_dir);
    fs::path runDir =
        run_dir.empty()
            ? sourcemaps::SourceMapStore::inferRunDir(treesDir)
            : fs::absolute(run_dir);

    if (raw &&
        (runDir.empty() ||
         !fs::exists(runDir / "source_maps"))) {
        throw std::runtime_error(
            "When raw=True, run_dir must contain source_maps/");
    }

    if (tree_ids.empty()) {
        tree_ids = discover_tree_ids(treesDir);
    }
    if (tree_ids.empty()) {
        throw std::runtime_error("No tree_*.bin found");
    }

    sourcemaps::SourceMapStore store(
        runDir, cols_per_shard);
    const std::vector<int> row_codes =
        encode_row_codes(store, row, raw);

    PredictDistribution predictor(treesDir.string());
    predictor.preload(tree_ids);

    ProbabilityRow psi(row_codes.size());
    std::unordered_set<int> learned(
        tree_ids.begin(), tree_ids.end());

    const int nt =
        threads > 0 ? threads : omp_get_max_threads();
    std::atomic<bool> failed{false};
    std::exception_ptr first_error;
    std::mutex error_mutex;

    {
        py::gil_scoped_release release;

        #pragma omp parallel for schedule(dynamic) num_threads(nt)
        for (ssize_t k = 0;
             k < static_cast<ssize_t>(tree_ids.size());
             ++k) {
            if (failed.load(std::memory_order_relaxed)) {
                continue;
            }

            try {
                const int tid =
                    tree_ids[static_cast<size_t>(k)];

                if (tid < 0 ||
                    static_cast<size_t>(tid) >=
                        row_codes.size()) {
                    continue;
                }

                std::vector<int> conditioned = row_codes;
                conditioned[static_cast<size_t>(tid)] = 0;

                psi[static_cast<size_t>(tid)] =
                    predictor.predictProbability(
                        tid, conditioned);
            } catch (...) {
                failed.store(
                    true, std::memory_order_relaxed);
                std::lock_guard<std::mutex> lock(
                    error_mutex);
                if (!first_error) {
                    first_error =
                        std::current_exception();
                }
            }
        }
    }

    if (first_error) {
        std::rethrow_exception(first_error);
    }

    for (size_t i = 0; i < psi.size(); ++i) {
        if (!psi[i].empty()) continue;

        const int code = row_codes[i];
        if (code != 0) {
            psi[i][code] = 1.0;
        } else if (!learned.count(
                       static_cast<int>(i))) {
            throw std::runtime_error(
                "Cannot initialize missing non-target column " +
                std::to_string(i) +
                " from a hard row");
        }
    }

    return std::make_unique<CenteredLdpPsiState>(
        trees_dir,
        std::move(psi),
        raw,
        run_dir,
        cols_per_shard,
        std::move(tree_ids));
}


static py::dict centered_ldp_converge_rows_py(
        const std::string& trees_dir,
        py::object rows_obj,
        int sweeps,
        int empirical_n,
        uint64_t seed,
        double response_scale,
        bool random_permutation,
        double tol,
        int patience,
        bool verbose,
        bool raw,
        std::string run_dir,
        int cols_per_shard,
        std::vector<int> tree_ids,
        int threads,
        int row_parallelism) {
    if (sweeps < 0) {
        throw std::invalid_argument("sweeps must be >= 0");
    }
    if (empirical_n <= 0) {
        throw std::invalid_argument("empirical_n must be > 0");
    }
    if (tol < 0.0) {
        throw std::invalid_argument("tol must be >= 0");
    }
    if (patience < 1) {
        throw std::invalid_argument("patience must be >= 1");
    }

    fs::path treesDir = fs::absolute(trees_dir);
    fs::path runDir =
        run_dir.empty()
            ? sourcemaps::SourceMapStore::inferRunDir(treesDir)
            : fs::absolute(run_dir);

    if (raw &&
        (runDir.empty() ||
         !fs::exists(runDir / "source_maps"))) {
        throw std::runtime_error(
            "When raw=True, run_dir must contain source_maps/");
    }

    if (tree_ids.empty()) {
        tree_ids = discover_tree_ids(treesDir);
    }
    if (tree_ids.empty()) {
        throw std::runtime_error("No tree_*.bin found");
    }

    py::sequence rows =
        py::reinterpret_borrow<py::sequence>(rows_obj);
    const ssize_t n_rows = py::len(rows);
    if (n_rows <= 0) {
        throw std::invalid_argument("rows must be non-empty");
    }

    sourcemaps::SourceMapStore store(
        runDir, cols_per_shard);

    std::vector<std::vector<int>> encoded(
        static_cast<size_t>(n_rows));
    size_t width = 0;

    for (ssize_t r = 0; r < n_rows; ++r) {
        py::sequence row =
            py::reinterpret_borrow<py::sequence>(rows[r]);

        if (r == 0) {
            width = static_cast<size_t>(py::len(row));
            if (width == 0) {
                throw std::invalid_argument(
                    "rows must have nonzero width");
            }
        } else if (
            static_cast<size_t>(py::len(row)) != width) {
            throw std::invalid_argument(
                "all rows must have the same width");
        }

        if (raw) {
            std::vector<std::string> tokens(width);
            for (size_t j = 0; j < width; ++j) {
                tokens[j] = py::str(row[j]);
            }
            encoded[static_cast<size_t>(r)] =
                store.encodeRow(tokens);
        } else {
            std::vector<int> codes(width, 0);
            for (size_t j = 0; j < width; ++j) {
                codes[j] = py::cast<int>(row[j]);
            }
            encoded[static_cast<size_t>(r)] =
                std::move(codes);
        }
    }

    const int total_threads =
        threads > 0 ? threads : omp_get_max_threads();
    const int outer_threads =
        std::max(
            1,
            std::min(
                static_cast<int>(n_rows),
                row_parallelism > 0
                    ? row_parallelism
                    : total_threads));

    // When multiple trajectories are running concurrently, keep each
    // trajectory single-threaded internally to avoid nested OpenMP
    // oversubscription.  A single-row batch can still use all threads.
    const int inner_threads =
        outer_threads > 1 ? 1 : total_threads;

    auto shared_predictor =
        std::make_shared<PredictDistribution>(
            treesDir.string());
    shared_predictor->preload(tree_ids);

    auto shared_cache =
        std::make_shared<CenteredHardKernelCache>();

    auto shared_dependencies =
        std::make_shared<std::vector<std::vector<int>>>(
            width);

    std::unordered_set<int> learned(
        tree_ids.begin(), tree_ids.end());

    for (int target : tree_ids) {
        const std::set<int> used =
            shared_predictor->usedColumns(target);
        for (int source : used) {
            if (source == target ||
                source < 0 ||
                static_cast<size_t>(source) >= width ||
                !learned.count(source)) {
                continue;
            }
            (*shared_dependencies)[
                static_cast<size_t>(source)]
                .push_back(target);
        }
    }

    for (auto& targets : *shared_dependencies) {
        std::sort(targets.begin(), targets.end());
        targets.erase(
            std::unique(targets.begin(), targets.end()),
            targets.end());
    }

    const auto batch_started =
        std::chrono::steady_clock::now();

    if (verbose) {
        std::cerr
            << "[centered_ldp] rows=" << n_rows
            << " sweeps=" << sweeps
            << " empirical_n=" << empirical_n
            << " total_threads=" << total_threads
            << " row_parallelism=" << outer_threads
            << " inner_threads=" << inner_threads
            << std::endl;
    }

    std::vector<std::vector<int>> hard_rows(
        static_cast<size_t>(n_rows));
    std::vector<int> sweeps_run(
        static_cast<size_t>(n_rows), 0);
    std::vector<double> final_mean_tv(
        static_cast<size_t>(n_rows), 0.0);
    std::vector<double> final_max_tv(
        static_cast<size_t>(n_rows), 0.0);
    std::vector<int> final_max_col(
        static_cast<size_t>(n_rows), -1);
    std::vector<double> final_event_tv(
        static_cast<size_t>(n_rows), 0.0);
    std::vector<double> final_sanov_exponent(
        static_cast<size_t>(n_rows), 0.0);
    std::vector<int> final_projected_count(
        static_cast<size_t>(n_rows), 0);

    {
        py::gil_scoped_release release;

        std::atomic<bool> failed{false};
        std::exception_ptr first_error;
        std::mutex error_mutex;
        std::mutex log_mutex;

        #pragma omp parallel for schedule(dynamic) num_threads(outer_threads)
        for (ssize_t r = 0; r < n_rows; ++r) {
            if (failed.load(std::memory_order_relaxed)) continue;

            try {
                const auto row_started =
                    std::chrono::steady_clock::now();

                if (verbose) {
                    std::lock_guard<std::mutex> lock(log_mutex);
                    std::cerr
                        << "[centered_ldp] row "
                        << (r + 1) << "/" << n_rows
                        << " initialize"
                        << std::endl;
                }

                ProbabilityRow placeholder(width);
                CenteredLdpPsiState state(
                    trees_dir,
                    std::move(placeholder),
                    raw,
                    run_dir,
                    cols_per_shard,
                    tree_ids,
                    shared_predictor,
                    shared_cache,
                    shared_dependencies);

                state.resetFromHardCodes(
                    encoded[static_cast<size_t>(r)],
                    inner_threads);

                if (verbose) {
                    std::lock_guard<std::mutex> lock(log_mutex);
                    std::cerr
                        << "[centered_ldp] row "
                        << (r + 1) << "/" << n_rows
                        << " initialized; cache="
                        << shared_cache->size()
                        << std::endl;
                }

                int quiet = 0;
                LdpStepSummary last;

                for (int si = 0; si < sweeps; ++si) {
                    const uint64_t sweep_seed = splitmix64(
                        (seed ? seed : 1ULL) ^
                        (static_cast<uint64_t>(r + 1) << 32) ^
                        static_cast<uint64_t>(si + 1));

                    last = state.sweep(
                        empirical_n,
                        "mode",
                        sweep_seed,
                        response_scale,
                        inner_threads,
                        random_permutation);

                    sweeps_run[static_cast<size_t>(r)] = si + 1;

                    if (verbose) {
                        std::lock_guard<std::mutex> lock(log_mutex);
                        std::cerr
                            << "[centered_ldp] row "
                            << (r + 1) << "/" << n_rows
                            << " sweep " << (si + 1)
                            << "/" << sweeps
                            << " meanTV=" << last.mean_tv
                            << " maxTV=" << last.max_tv
                            << " cache=" << shared_cache->size()
                            << std::endl;
                    }

                    if (tol > 0.0 && last.max_tv <= tol) {
                        ++quiet;
                        if (quiet >= patience) break;
                    } else {
                        quiet = 0;
                    }
                }

                hard_rows[static_cast<size_t>(r)] =
                    state.hardCodes();

                if (sweeps_run[static_cast<size_t>(r)] > 0) {
                    final_mean_tv[static_cast<size_t>(r)] =
                        last.mean_tv;
                    final_max_tv[static_cast<size_t>(r)] =
                        last.max_tv;
                    final_max_col[static_cast<size_t>(r)] =
                        last.max_col;
                    final_event_tv[static_cast<size_t>(r)] =
                        last.event_tv;
                    final_sanov_exponent[static_cast<size_t>(r)] =
                        last.sanov_exponent;
                    final_projected_count[static_cast<size_t>(r)] =
                        last.projected_count;
                }

                if (verbose) {
                    const auto row_done =
                        std::chrono::steady_clock::now();
                    const double row_sec =
                        std::chrono::duration<double>(
                            row_done - row_started).count();

                    std::lock_guard<std::mutex> lock(log_mutex);
                    std::cerr
                        << "[centered_ldp] row "
                        << (r + 1) << "/" << n_rows
                        << " done sec=" << row_sec
                        << " sweeps="
                        << sweeps_run[static_cast<size_t>(r)]
                        << " final_meanTV="
                        << final_mean_tv[static_cast<size_t>(r)]
                        << " final_maxTV="
                        << final_max_tv[static_cast<size_t>(r)]
                        << " cache=" << shared_cache->size()
                        << std::endl;
                }
            } catch (...) {
                failed.store(true, std::memory_order_relaxed);
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!first_error) {
                    first_error = std::current_exception();
                }
            }
        }

        if (first_error) {
            std::rethrow_exception(first_error);
        }
    }

    if (verbose) {
        const auto batch_done =
            std::chrono::steady_clock::now();
        const double batch_sec =
            std::chrono::duration<double>(
                batch_done - batch_started).count();
        std::cerr
            << "[centered_ldp] complete sec="
            << batch_sec
            << " cache=" << shared_cache->size()
            << std::endl;
    }

    py::list out_rows;
    for (ssize_t r = 0; r < n_rows; ++r) {
        py::list row;
        const auto& codes =
            hard_rows[static_cast<size_t>(r)];

        for (size_t j = 0; j < codes.size(); ++j) {
            const int code = codes[j];
            if (raw) {
                if (code == 0) {
                    row.append(py::str(""));
                } else if (store.tryGet(static_cast<int>(j))) {
                    row.append(py::str(
                        store.decodeLabel(
                            static_cast<int>(j), code)));
                } else {
                    row.append(py::str(std::to_string(code)));
                }
            } else {
                row.append(py::int_(code));
            }
        }
        out_rows.append(std::move(row));
    }

    py::dict out;
    out["hard_rows"] = std::move(out_rows);
    out["sweeps_run"] = sweeps_run;
    out["final_mean_tv"] = final_mean_tv;
    out["final_max_tv"] = final_max_tv;
    out["final_max_col"] = final_max_col;
    out["final_event_tv"] = final_event_tv;
    out["final_sanov_exponent"] = final_sanov_exponent;
    out["final_projected_count"] = final_projected_count;
    out["kernel_cache_size"] = shared_cache->size();
    out["sweeps_requested"] = sweeps;
    out["empirical_n"] = empirical_n;
    out["row_parallelism"] = outer_threads;
    out["inner_threads"] = inner_threads;
    return out;
}

static py::dict step_summary_to_py(const PsiStepSummary& s) {
    py::dict out;
    out["step"] = s.step;
    out["mean_tv"] = s.mean_tv;
    out["max_tv"] = s.max_tv;
    out["max_col"] = s.max_col;
    return out;
}

static py::list phi_py(
        const std::string& trees_dir,
        py::object psi_obj,
        bool raw,
        std::string run_dir,
        int cols_per_shard,
        std::vector<int> tree_ids,
        int threads) {
    fs::path treesDir = fs::absolute(trees_dir);
    fs::path runDir = run_dir.empty()
                          ? sourcemaps::SourceMapStore::inferRunDir(treesDir)
                          : fs::absolute(run_dir);
    if (raw && (runDir.empty() || !fs::exists(runDir / "source_maps"))) {
        throw std::runtime_error("When raw=True, run_dir must contain source_maps/");
    }
    if (tree_ids.empty()) tree_ids = discover_tree_ids(treesDir);
    if (tree_ids.empty()) throw std::runtime_error("No tree_*.bin found");

    sourcemaps::SourceMapStore store(runDir, cols_per_shard);
    const ProbabilityRow psi = python_to_psi(store, psi_obj, raw);
    PredictDistribution predictor(treesDir.string());

    ProbabilityRow next;
    {
        py::gil_scoped_release release;
        next = phi_native(predictor, psi, tree_ids, threads);
    }
    return psi_to_python(store, next, raw);
}



struct EmptyTowerPairResult {
    int source = -1;
    int target = -1;
    int source_support = 0;
    double tv = 0.0;
};

static ProbabilityDistribution weighted_single_observation_response(
        PredictDistribution& predictor,
        int source,
        int target,
        const ProbabilityDistribution& source_marginal,
        size_t width) {
    ProbabilityDistribution mixed;
    std::vector<int> row(width, 0);

    for (const auto& kv : source_marginal) {
        const int sigma = kv.first;
        const double weight = kv.second;
        if (!(weight > 0.0)) continue;

        row[static_cast<size_t>(source)] = sigma;
        const ProbabilityDistribution q =
            predictor.predictProbability(target, row);
        for (const auto& out : q) {
            mixed[out.first] += weight * out.second;
        }
        row[static_cast<size_t>(source)] = 0;
    }

    double total = 0.0;
    for (const auto& kv : mixed) total += kv.second;
    if (!(total > 0.0)) {
        throw std::logic_error(
            "empty_tower_check: weighted response has zero mass");
    }
    for (auto& kv : mixed) kv.second /= total;
    return mixed;
}

static double empirical_quantile(
        const std::vector<double>& sorted,
        double q) {
    if (sorted.empty()) return 0.0;
    if (q <= 0.0) return sorted.front();
    if (q >= 1.0) return sorted.back();
    const double pos = q * static_cast<double>(sorted.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = static_cast<size_t>(std::ceil(pos));
    if (lo == hi) return sorted[lo];
    const double w = pos - static_cast<double>(lo);
    return (1.0 - w) * sorted[lo] + w * sorted[hi];
}

static py::dict empty_tower_check_py(
        const std::string& trees_dir,
        int top_k,
        bool raw,
        std::string run_dir,
        int cols_per_shard,
        std::vector<int> tree_ids,
        int threads) {
    fs::path treesDir = fs::absolute(trees_dir);
    fs::path runDir = run_dir.empty()
                          ? sourcemaps::SourceMapStore::inferRunDir(treesDir)
                          : fs::absolute(run_dir);
    if (raw && (runDir.empty() || !fs::exists(runDir / "source_maps"))) {
        throw std::runtime_error(
            "When raw=True, run_dir must contain source_maps/");
    }
    if (tree_ids.empty()) tree_ids = discover_tree_ids(treesDir);
    if (tree_ids.empty()) throw std::runtime_error("No tree_*.bin found");
    if (top_k < 0) throw std::invalid_argument("top_k must be >= 0");

    const int max_id = *std::max_element(tree_ids.begin(), tree_ids.end());
    const size_t width = static_cast<size_t>(max_id + 1);
    std::unordered_set<int> learned(tree_ids.begin(), tree_ids.end());

    sourcemaps::SourceMapStore store(runDir, cols_per_shard);
    PredictDistribution predictor(treesDir.string());
    predictor.preload(tree_ids);

    // Empty-sample marginals p_i = phi_i(empty).
    ProbabilityRow p0(width);
    std::vector<int> empty(width, 0);

    const int nt = threads > 0 ? threads : omp_get_max_threads();
    std::atomic<bool> failed{false};
    std::exception_ptr first_error;
    std::mutex error_mutex;

    #pragma omp parallel for schedule(dynamic) num_threads(nt)
    for (ssize_t k = 0; k < static_cast<ssize_t>(tree_ids.size()); ++k) {
        if (failed.load(std::memory_order_relaxed)) continue;
        try {
            const int tid = tree_ids[static_cast<size_t>(k)];
            p0[static_cast<size_t>(tid)] =
                predictor.predictProbability(tid, empty);
        } catch (...) {
            failed.store(true, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lock(error_mutex);
            if (!first_error) first_error = std::current_exception();
        }
    }
    if (first_error) std::rethrow_exception(first_error);

    // Only pairs i->j where tree j actually uses i can be non-trivial.
    std::vector<std::pair<int,int>> tasks;
    for (int target : tree_ids) {
        const std::set<int> used = predictor.usedColumns(target);
        for (int source : used) {
            if (source == target || !learned.count(source)) continue;
            tasks.emplace_back(source, target);
        }
    }

    std::vector<EmptyTowerPairResult> results(tasks.size());
    failed.store(false, std::memory_order_relaxed);
    first_error = nullptr;

    #pragma omp parallel for schedule(dynamic) num_threads(nt)
    for (ssize_t k = 0; k < static_cast<ssize_t>(tasks.size()); ++k) {
        if (failed.load(std::memory_order_relaxed)) continue;
        try {
            const int source = tasks[static_cast<size_t>(k)].first;
            const int target = tasks[static_cast<size_t>(k)].second;

            const ProbabilityDistribution mixed =
                weighted_single_observation_response(
                    predictor,
                    source,
                    target,
                    p0[static_cast<size_t>(source)],
                    width);

            EmptyTowerPairResult r;
            r.source = source;
            r.target = target;
            r.source_support =
                static_cast<int>(p0[static_cast<size_t>(source)].size());
            r.tv = tv_distance(
                mixed, p0[static_cast<size_t>(target)]);
            results[static_cast<size_t>(k)] = r;
        } catch (...) {
            failed.store(true, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lock(error_mutex);
            if (!first_error) first_error = std::current_exception();
        }
    }
    if (first_error) std::rethrow_exception(first_error);

    std::vector<double> tvs;
    tvs.reserve(results.size());
    double sum_tv = 0.0;
    size_t gt_1e12 = 0, gt_1e9 = 0, gt_1e6 = 0;
    size_t gt_1e4 = 0, gt_1e3 = 0, gt_1e2 = 0, gt_5e2 = 0;

    for (const auto& r : results) {
        tvs.push_back(r.tv);
        sum_tv += r.tv;
        if (r.tv > 1e-12) ++gt_1e12;
        if (r.tv > 1e-9)  ++gt_1e9;
        if (r.tv > 1e-6)  ++gt_1e6;
        if (r.tv > 1e-4)  ++gt_1e4;
        if (r.tv > 1e-3)  ++gt_1e3;
        if (r.tv > 1e-2)  ++gt_1e2;
        if (r.tv > 5e-2)  ++gt_5e2;
    }
    std::sort(tvs.begin(), tvs.end());

    std::sort(
        results.begin(), results.end(),
        [](const EmptyTowerPairResult& a,
           const EmptyTowerPairResult& b) {
            if (a.tv != b.tv) return a.tv > b.tv;
            if (a.target != b.target) return a.target < b.target;
            return a.source < b.source;
        });

    const size_t d = tree_ids.size();
    const size_t total_pairs = d > 0 ? d * (d - 1) : 0;
    const size_t dependency_pairs = results.size();
    const size_t structural_zero_pairs =
        total_pairs >= dependency_pairs
            ? total_pairs - dependency_pairs
            : 0;

    py::dict out;
    out["variables"] = d;
    out["ordered_pairs"] = total_pairs;
    out["dependency_pairs"] = dependency_pairs;
    out["structural_zero_pairs"] = structural_zero_pairs;
    out["mean_tv_dependency"] =
        dependency_pairs
            ? sum_tv / static_cast<double>(dependency_pairs)
            : 0.0;
    out["mean_tv_all_pairs"] =
        total_pairs
            ? sum_tv / static_cast<double>(total_pairs)
            : 0.0;
    out["median_tv_dependency"] = empirical_quantile(tvs, 0.50);
    out["p90_tv_dependency"] = empirical_quantile(tvs, 0.90);
    out["p95_tv_dependency"] = empirical_quantile(tvs, 0.95);
    out["p99_tv_dependency"] = empirical_quantile(tvs, 0.99);
    out["max_tv_dependency"] =
        tvs.empty() ? 0.0 : tvs.back();

    py::dict thresholds;
    thresholds[">1e-12"] = gt_1e12;
    thresholds[">1e-9"] = gt_1e9;
    thresholds[">1e-6"] = gt_1e6;
    thresholds[">1e-4"] = gt_1e4;
    thresholds[">1e-3"] = gt_1e3;
    thresholds[">1e-2"] = gt_1e2;
    thresholds[">5e-2"] = gt_5e2;
    out["counts_above"] = thresholds;

    py::list top;
    const size_t n_top =
        std::min<size_t>(
            static_cast<size_t>(top_k), results.size());

    // Recompute only the few worst pairs to expose the actual two sides.
    for (size_t k = 0; k < n_top; ++k) {
        const auto& r = results[k];
        const ProbabilityDistribution mixed =
            weighted_single_observation_response(
                predictor,
                r.source,
                r.target,
                p0[static_cast<size_t>(r.source)],
                width);

        py::dict item;
        item["source"] = r.source;
        item["target"] = r.target;
        item["source_support"] = r.source_support;
        item["tv"] = r.tv;
        item["weighted"] = distribution_to_python(
            store, r.target, mixed, raw);
        item["empty_target"] = distribution_to_python(
            store, r.target,
            p0[static_cast<size_t>(r.target)], raw);
        top.append(item);
    }
    out["top"] = top;

    return out;
}

static py::dict ldp_update_py(
        const std::string& trees_dir,
        py::object psi_obj,
        int target_col,
        int n,
        const std::string& event,
        uint64_t seed,
        py::object empirical_obj,
        double response_scale,
        bool raw,
        std::string run_dir,
        int cols_per_shard,
        std::vector<int> tree_ids,
        int threads) {
    fs::path treesDir = fs::absolute(trees_dir);
    fs::path runDir = run_dir.empty()
                          ? sourcemaps::SourceMapStore::inferRunDir(treesDir)
                          : fs::absolute(run_dir);
    if (raw && (runDir.empty() || !fs::exists(runDir / "source_maps"))) {
        throw std::runtime_error(
            "When raw=True, run_dir must contain source_maps/");
    }
    if (tree_ids.empty()) tree_ids = discover_tree_ids(treesDir);
    if (tree_ids.empty()) throw std::runtime_error("No tree_*.bin found");
    if (!(response_scale >= 0.0) || !std::isfinite(response_scale)) {
        throw std::invalid_argument(
            "response_scale must be finite and non-negative");
    }

    sourcemaps::SourceMapStore store(runDir, cols_per_shard);
    const ProbabilityRow psi = python_to_psi(store, psi_obj, raw);
    if (target_col < 0) {
        std::mt19937_64 rng(seed ? seed : 1ULL);
        std::uniform_int_distribution<size_t> pick(0, tree_ids.size() - 1);
        target_col = tree_ids[pick(rng)];
    }
    if (target_col < 0 || static_cast<size_t>(target_col) >= psi.size()) {
        throw std::out_of_range("target_col out of range");
    }
    if (std::find(tree_ids.begin(), tree_ids.end(), target_col) ==
        tree_ids.end()) {
        throw std::invalid_argument("target_col has no learned tree");
    }

    ProbabilityDistribution empirical;
    std::string event_mode;
    if (!empirical_obj.is_none()) {
        empirical = python_to_distribution(
            store, target_col, empirical_obj, raw);
        event_mode = "supplied";
    } else if (event == "zero_action" || n == 0) {
        empirical = psi[static_cast<size_t>(target_col)];
        event_mode = "zero_action";
    } else if (event == "mode") {
        empirical = multinomial_mode_empirical(
            psi[static_cast<size_t>(target_col)], n);
        event_mode = "mode";
    } else if (event == "sample") {
        empirical = multinomial_empirical(
            psi[static_cast<size_t>(target_col)], n, seed);
        event_mode = "sample";
    } else {
        throw std::invalid_argument(
            "event must be 'zero_action', 'mode', or 'sample'");
    }

    ProbabilityRow perturbed_state = psi;
    perturbed_state[static_cast<size_t>(target_col)] = empirical;

    PredictDistribution predictor(treesDir.string());
    ProbabilityRow base_response;
    ProbabilityRow pert_response;
    {
        py::gil_scoped_release release;
        base_response = phi_native(
            predictor, psi, tree_ids, threads);
        pert_response = phi_native(
            predictor, perturbed_state, tree_ids, threads);
    }

    ProbabilityRow next = psi;
    next[static_cast<size_t>(target_col)] = empirical;

    py::list projected_cols;
    for (int tid : tree_ids) {
        if (tid == target_col || tid < 0 ||
            static_cast<size_t>(tid) >= psi.size()) continue;
        bool projected = false;
        next[static_cast<size_t>(tid)] = add_response_delta(
            psi[static_cast<size_t>(tid)],
            base_response[static_cast<size_t>(tid)],
            pert_response[static_cast<size_t>(tid)],
            response_scale,
            &projected);
        if (projected) projected_cols.append(tid);
    }

    const PsiStepSummary summary =
        summarize_step(psi, next, tree_ids, 1);

    const double event_tv = tv_distance(
        psi[static_cast<size_t>(target_col)], empirical);

    py::dict out;
    out["psi"] = psi_to_python(store, next, raw);
    out["target_col"] = target_col;
    out["event_mode"] = event_mode;
    out["empirical"] = distribution_to_python(
        store, target_col, empirical, raw);
    out["event_tv"] = event_tv;
    const double sanov_rate = kl_divergence(
        empirical, psi[static_cast<size_t>(target_col)]);
    out["sanov_rate"] = sanov_rate;
    if (n > 0 && std::isfinite(sanov_rate)) {
        out["sanov_exponent"] =
            static_cast<double>(n) * sanov_rate;
    } else {
        out["sanov_exponent"] = py::none();
    }
    out["mean_tv"] = summary.mean_tv;
    out["max_tv"] = summary.max_tv;
    out["max_col"] = summary.max_col;
    out["projected_cols"] = projected_cols;
    out["response_scale"] = response_scale;
    if (event_mode == "sample" || event_mode == "mode") out["n"] = n;
    else out["n"] = py::none();
    return out;
}

static std::unique_ptr<PsiState> psi_state_from_row_py(
        const std::string& trees_dir,
        py::array row,
        bool raw,
        std::string run_dir,
        int cols_per_shard,
        std::vector<int> tree_ids) {
    fs::path treesDir = fs::absolute(trees_dir);
    fs::path runDir = run_dir.empty()
                          ? sourcemaps::SourceMapStore::inferRunDir(treesDir)
                          : fs::absolute(run_dir);
    if (raw && (runDir.empty() || !fs::exists(runDir / "source_maps"))) {
        throw std::runtime_error("When raw=True, run_dir must contain source_maps/");
    }
    if (tree_ids.empty()) tree_ids = discover_tree_ids(treesDir);
    if (tree_ids.empty()) throw std::runtime_error("No tree_*.bin found");

    sourcemaps::SourceMapStore store(runDir, cols_per_shard);
    const std::vector<int> row_codes = encode_row_codes(store, row, raw);
    std::unordered_set<int> available(tree_ids.begin(), tree_ids.end());
    PredictDistribution predictor(treesDir.string());

    ProbabilityRow psi(row_codes.size());
    for (size_t i = 0; i < row_codes.size(); ++i) {
        const int code = row_codes[i];
        if (code != 0) {
            psi[i][code] = 1.0;
        } else {
            if (!available.count(static_cast<int>(i))) {
                throw std::runtime_error(
                    "missing column has no learned tree: " + std::to_string(i));
            }
            psi[i] = counts_to_distribution(
                predictor.predict(static_cast<int>(i), row_codes));
        }
    }

    return std::make_unique<PsiState>(
        trees_dir, std::move(psi), raw, run_dir,
        cols_per_shard, std::move(tree_ids));
}

static py::list omega_py(const std::string& trees_dir, py::array row, bool raw, std::string run_dir, int cols_per_shard, std::vector<int> tree_ids) {
    fs::path treesDir = fs::absolute(trees_dir);
    fs::path runDir = run_dir.empty() ? sourcemaps::SourceMapStore::inferRunDir(treesDir) : fs::absolute(run_dir);

    if (raw && (runDir.empty() || !fs::exists(runDir / "source_maps"))) {
        throw std::runtime_error("When raw=True, run_dir must contain source_maps/");
    }

    if(tree_ids.empty()) tree_ids = discover_tree_ids(treesDir);
    if(tree_ids.empty()) throw std::runtime_error("No tree_*.bin found and tree_ids is empty");

    sourcemaps::SourceMapStore store(runDir, cols_per_shard);
    
    py::sequence seq = py::reinterpret_borrow<py::sequence>(row);
    const ssize_t num_tokens = py::len(seq);
    std::vector<std::string> out(static_cast<size_t>(num_tokens));
    for(ssize_t i=0; i< num_tokens; ++i) {
        out[static_cast<size_t>(i)] = py::str(seq[i]);
    }

    const std::vector<int> row_codes = encode_row_codes(store, row, raw);
    PredictDistribution pd(treesDir.string());

    for(int tid : tree_ids) {
        if(tid <0 || tid >= num_tokens) continue;
        auto counts = pd.predict(tid, row_codes);
        if(counts.empty()) continue;
        out[static_cast<size_t>(tid)] = argmax_label(store, tid, counts);
    }

    py::list result(num_tokens);
    for (ssize_t i = 0; i < num_tokens; ++i) {
        result[i] = py::str(out[static_cast<size_t>(i)]);
    }
    return result;
}

static py::array_t<double> normalized_profiles_batch_py(
        const std::string& trees_dir, py::object rows, std::string run_dir,
        int cols_per_shard, std::vector<int> tree_ids, int threads, double eps_floor) {
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
    size_t p = 0;
    for (ssize_t i = 0; i < n; ++i) {
        py::sequence row = py::reinterpret_borrow<py::sequence>(seq[i]);
        if (i == 0) p = static_cast<size_t>(py::len(row));
        std::vector<std::string> tok(p);
        if (static_cast<size_t>(py::len(row)) != p) throw std::invalid_argument("row widths differ");
        for (size_t j = 0; j < p; ++j) tok[j] = py::str(row[j]);
        codes[static_cast<size_t>(i)] = store.encodeRow(tok);
    }

    PredictDistribution predictor(treesDir.string());
    predictor.preload(tree_ids);
    auto out = py::array_t<double>({n, static_cast<ssize_t>(p)});
    auto a = out.mutable_unchecked<2>();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (ssize_t i = 0; i < n; ++i) for (size_t j = 0; j < p; ++j) a(i, j) = nan;
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
                const auto& row = codes[static_cast<size_t>(i)];
                for (int tid : tree_ids) {
                    if (tid < 0 || static_cast<size_t>(tid) >= p) continue;
                    const int observed = row[static_cast<size_t>(tid)];
                    if (observed == 0) continue;
                    const auto counts = predictor.predict(tid, row);
                    long long total = 0, maximum = 0, hit = 0;
                    for (const auto& kv : counts) {
                        total += kv.second;
                        maximum = std::max(maximum, static_cast<long long>(kv.second));
                        if (kv.first == observed) hit += kv.second;
                    }
                    if (total <= 0 || maximum <= 0) continue;
                    double pobs = static_cast<double>(hit) / static_cast<double>(total);
                    pobs = std::max(pobs, eps_floor);
                    const double pmax = static_cast<double>(maximum) / static_cast<double>(total);
                    a(i, tid) = pobs / pmax;
                }
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


static py::dict pseudo_code_lengths_batch_py(
        const std::string& trees_dir, py::object rows, std::string run_dir,
        int cols_per_shard, std::vector<int> tree_ids, int threads,
        double prob_floor, bool return_per_col) {
    if (!(prob_floor > 0.0 && prob_floor <= 1.0)) {
        throw std::invalid_argument("prob_floor must be in (0,1]");
    }

    fs::path treesDir = fs::absolute(trees_dir);
    fs::path runDir = run_dir.empty()
                          ? sourcemaps::SourceMapStore::inferRunDir(treesDir)
                          : fs::absolute(run_dir);
    if (runDir.empty() || !fs::exists(runDir / "source_maps")) {
        throw std::runtime_error("run_dir must contain source_maps/");
    }
    if (tree_ids.empty()) tree_ids = discover_tree_ids(treesDir);
    if (tree_ids.empty()) throw std::runtime_error("no trees found");

    // Encode while holding the GIL. SourceMapStore is not touched by worker threads.
    py::sequence seq = py::reinterpret_borrow<py::sequence>(rows);
    const ssize_t n = py::len(seq);
    if (n <= 0) throw std::invalid_argument("rows must be non-empty");
    sourcemaps::SourceMapStore store(runDir, cols_per_shard);
    std::vector<std::vector<int>> codes(static_cast<size_t>(n));
    size_t p = 0;
    for (ssize_t i = 0; i < n; ++i) {
        py::sequence row = py::reinterpret_borrow<py::sequence>(seq[i]);
        if (i == 0) p = static_cast<size_t>(py::len(row));
        if (static_cast<size_t>(py::len(row)) != p) {
            throw std::invalid_argument("row widths differ");
        }
        std::vector<std::string> tok(p);
        for (size_t j = 0; j < p; ++j) tok[j] = py::str(row[j]);
        codes[static_cast<size_t>(i)] = store.encodeRow(tok);
    }

    PredictDistribution predictor(treesDir.string());
    predictor.preload(tree_ids);

    std::vector<double> lengths(static_cast<size_t>(n), 0.0);
    std::vector<int> used(static_cast<size_t>(n), 0);
    std::vector<double> per_col;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (return_per_col) per_col.assign(static_cast<size_t>(n) * p, nan);

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
                const auto& row = codes[static_cast<size_t>(i)];
                double bits = 0.0;
                int n_used = 0;
                for (int tid : tree_ids) {
                    if (tid < 0 || static_cast<size_t>(tid) >= p) continue;
                    const int observed = row[static_cast<size_t>(tid)];
                    if (observed == 0) continue;

                    const auto counts = predictor.predict(tid, row);
                    long long total = 0;
                    long long hit = 0;
                    for (const auto& kv : counts) {
                        total += kv.second;
                        if (kv.first == observed) hit += kv.second;
                    }
                    if (total <= 0) continue;

                    const double pobs = std::max(
                        static_cast<double>(hit) / static_cast<double>(total),
                        prob_floor);
                    const double b = -std::log2(pobs);
                    bits += b;
                    ++n_used;
                    if (return_per_col) {
                        per_col[static_cast<size_t>(i) * p + static_cast<size_t>(tid)] = b;
                    }
                }
                lengths[static_cast<size_t>(i)] = bits;
                used[static_cast<size_t>(i)] = n_used;
            } catch (...) {
                failed.store(true, std::memory_order_relaxed);
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!first_error) first_error = std::current_exception();
            }
        }
    }
    if (first_error) std::rethrow_exception(first_error);

    py::array_t<double> out_lengths(n);
    py::array_t<int> out_used(n);
    auto l = out_lengths.mutable_unchecked<1>();
    auto u = out_used.mutable_unchecked<1>();
    for (ssize_t i = 0; i < n; ++i) {
        l(i) = lengths[static_cast<size_t>(i)];
        u(i) = used[static_cast<size_t>(i)];
    }

    py::dict out;
    out["bits"] = std::move(out_lengths);
    out["n_used"] = std::move(out_used);
    if (return_per_col) {
        py::array_t<double> pc({n, static_cast<ssize_t>(p)});
        auto a = pc.mutable_unchecked<2>();
        for (ssize_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < p; ++j) {
                a(i, static_cast<ssize_t>(j)) =
                    per_col[static_cast<size_t>(i) * p + j];
            }
        }
        out["per_col_bits"] = std::move(pc);
    }
    return out;
}

static py::dict persistence_batch_py(
        const std::string& trees_dir, py::object rows, std::string run_dir,
        int cols_per_shard, std::vector<int> tree_ids, int threads,
        double prob_floor) {
    if (!(prob_floor > 0.0 && prob_floor <= 1.0)) {
        throw std::invalid_argument("prob_floor must be in (0,1]");
    }
    fs::path treesDir = fs::absolute(trees_dir);
    fs::path runDir = run_dir.empty()
                          ? sourcemaps::SourceMapStore::inferRunDir(treesDir)
                          : fs::absolute(run_dir);
    if (runDir.empty() || !fs::exists(runDir / "source_maps")) {
        throw std::runtime_error("run_dir must contain source_maps/");
    }
    if (tree_ids.empty()) tree_ids = discover_tree_ids(treesDir);
    if (tree_ids.empty()) throw std::runtime_error("no trees found");

    py::sequence seq = py::reinterpret_borrow<py::sequence>(rows);
    const ssize_t n = py::len(seq);
    if (n <= 0) throw std::invalid_argument("rows must be non-empty");
    sourcemaps::SourceMapStore store(runDir, cols_per_shard);
    std::vector<std::vector<int>> codes(static_cast<size_t>(n));
    size_t p = 0;
    for (ssize_t i = 0; i < n; ++i) {
        py::sequence row = py::reinterpret_borrow<py::sequence>(seq[i]);
        if (i == 0) p = static_cast<size_t>(py::len(row));
        if (static_cast<size_t>(py::len(row)) != p) throw std::invalid_argument("row widths differ");
        std::vector<std::string> tok(p);
        for (size_t j = 0; j < p; ++j) tok[j] = py::str(row[j]);
        codes[static_cast<size_t>(i)] = store.encodeRow(tok);
    }

    PredictDistribution predictor(treesDir.string());
    predictor.preload(tree_ids);
    std::vector<double> persistence(static_cast<size_t>(n), 0.0);
    std::vector<int> used(static_cast<size_t>(n), 0);
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
                const auto& row = codes[static_cast<size_t>(i)];
                double total_logp = 0.0;
                int n_used = 0;
                for (int tid : tree_ids) {
                    if (tid < 0 || static_cast<size_t>(tid) >= p) continue;
                    const int observed = row[static_cast<size_t>(tid)];
                    if (observed == 0) continue;
                    const auto counts = predictor.predict(tid, row);
                    long long total = 0, hit = 0;
                    for (const auto& kv : counts) {
                        total += kv.second;
                        if (kv.first == observed) hit += kv.second;
                    }
                    if (total <= 0) continue;
                    const double pobs = std::max(
                        static_cast<double>(hit) / static_cast<double>(total),
                        prob_floor);
                    total_logp += std::log2(pobs);
                    ++n_used;
                }
                persistence[static_cast<size_t>(i)] = total_logp;
                used[static_cast<size_t>(i)] = n_used;
            } catch (...) {
                failed.store(true, std::memory_order_relaxed);
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!first_error) first_error = std::current_exception();
            }
        }
    }
    if (first_error) std::rethrow_exception(first_error);

    py::array_t<double> out_p(n);
    py::array_t<int> out_u(n);
    auto po = out_p.mutable_unchecked<1>();
    auto uo = out_u.mutable_unchecked<1>();
    for (ssize_t i = 0; i < n; ++i) {
        po(i) = persistence[static_cast<size_t>(i)];
        uo(i) = used[static_cast<size_t>(i)];
    }
    py::dict out;
    out["persistence"] = std::move(out_p);
    out["n_used"] = std::move(out_u);
    return out;
}

static py::dict persistence_py(const std::string& trees_dir,
                               py::array row,
                               bool raw,
                               std::string run_dir,
                               int cols_per_shard,
                               std::vector<int> tree_ids,
                               bool return_per_col) {
    fs::path treesDir = fs::absolute(trees_dir);
    fs::path runDir = run_dir.empty()
                          ? sourcemaps::SourceMapStore::inferRunDir(treesDir)
                          : fs::absolute(run_dir);

    if (raw && (runDir.empty() || !fs::exists(runDir / "source_maps"))) {
        throw std::runtime_error("When raw=True, run_dir must contain source_maps/");
    }

    if (tree_ids.empty()) tree_ids = discover_tree_ids(treesDir);
    if (tree_ids.empty()) throw std::runtime_error("No tree_*.bin found");

    sourcemaps::SourceMapStore store(runDir, cols_per_shard);
    py::sequence seq = py::reinterpret_borrow<py::sequence>(row);
    const ssize_t n = py::len(seq);
    std::vector<std::string> tokens(static_cast<size_t>(n));
    for (ssize_t i = 0; i < n; ++i) tokens[static_cast<size_t>(i)] = py::str(seq[i]);

    const std::vector<int> codes = encode_row_codes(store, row, raw);
    PredictDistribution pd(treesDir.string());

    double total = 0.0;
    std::size_t used = 0;
    py::list per;
    if (return_per_col) {
        for (ssize_t i = 0; i < n; ++i)
            per.append(py::float_(std::numeric_limits<double>::quiet_NaN()));
    }

    for (int tid : tree_ids) {
        if (tid < 0 || tid >= n) continue;
        auto counts = pd.predict(tid, codes);
        const double v = token_log2_prob(store, tid, counts, tokens[static_cast<size_t>(tid)]);
        if (return_per_col) per[tid] = py::float_(v);
        if (std::isfinite(v)) {
            total += v;
            ++used;
        }
    }

    py::dict out;
    out["persistence"] = total;
    out["n_used"] = static_cast<int>(used);
    if (return_per_col) out["per_col"] = std::move(per);
    return out;
}

PYBIND11_MODULE(predict_distribution, m) {
    py::class_<PsiState, std::unique_ptr<PsiState>>(m, "PsiState")
        .def("step",
             [](PsiState& self, int threads) {
                 PsiStepSummary summary;
                 {
                     py::gil_scoped_release release;
                     summary = self.step(threads);
                 }
                 return step_summary_to_py(summary);
             },
             py::arg("threads") = 0,
             "Advance one synchronous mean-field Phi step in resident C++ state.")
        .def("iterate",
             [](PsiState& self, int steps, int threads, double tol) {
                 std::vector<PsiStepSummary> history;
                 {
                     py::gil_scoped_release release;
                     history = self.iterate(steps, threads, tol);
                 }
                 py::list out;
                 for (const auto& h : history) out.append(step_summary_to_py(h));
                 return out;
             },
             py::arg("steps"),
             py::arg("threads") = 0,
             py::arg("tol") = 0.0,
             "Iterate synchronous mean-field Phi entirely in C++.")
        .def("to_python",
             [](PsiState& self) {
                 return psi_to_python(self.store(), self.psi(), self.raw());
             })
        .def("distribution",
             [](PsiState& self, int col) {
                 if (col < 0 || static_cast<size_t>(col) >= self.psi().size()) {
                     throw std::out_of_range("column out of range");
                 }
                 return distribution_to_python(
                     self.store(), col, self.psi()[static_cast<size_t>(col)],
                     self.raw());
             },
             py::arg("col"))
        .def_property_readonly("step_count", &PsiState::stepCount)
        .def_property_readonly("size",
             [](const PsiState& self) { return self.psi().size(); });

    m.def("psi_state_from_row",
          &psi_state_from_row_py,
          py::arg("trees_dir"),
          py::arg("row"),
          py::arg("raw") = true,
          py::arg("run_dir") = "",
          py::arg("cols_per_shard") = 50000,
          py::arg("tree_ids") = std::vector<int>{},
          "Create a C++-resident Psi state from a hard/partial row.");






    m.def("centered_ldp_converge_rows",
          &centered_ldp_converge_rows_py,
          py::arg("trees_dir"),
          py::arg("rows"),
          py::arg("sweeps") = 10,
          py::arg("empirical_n") = 10,
          py::arg("seed") = 1,
          py::arg("response_scale") = 1.0,
          py::arg("random_permutation") = false,
          py::arg("tol") = 0.0,
          py::arg("patience") = 1,
          py::arg("verbose") = false,
          py::arg("raw") = true,
          py::arg("run_dir") = "",
          py::arg("cols_per_shard") = 50000,
          py::arg("tree_ids") = std::vector<int>{},
          py::arg("threads") = 0,
          py::arg("row_parallelism") = 0,
          "Run centered hard-response LDP dynamics on a batch of hard rows "
          "using one resident model and a shared warmed response-kernel cache. "
          "Returns near-converged argmax hard rows and per-row residuals.");

    py::class_<
        CenteredLdpPsiState,
        std::unique_ptr<CenteredLdpPsiState>>(
            m, "CenteredLdpPsiState")
        .def("step",
             [](CenteredLdpPsiState& self,
                int source,
                int n,
                const std::string& event,
                uint64_t seed,
                double response_scale,
                int threads) {
                 LdpStepSummary summary;
                 {
                     py::gil_scoped_release release;
                     summary = self.step(
                         source,
                         n,
                         event,
                         seed,
                         response_scale,
                         threads);
                 }
                 return ldp_step_summary_to_py(summary);
             },
             py::arg("source") = -1,
             py::arg("n") = 10,
             py::arg("event") = "mode",
             py::arg("seed") = 1,
             py::arg("response_scale") = 1.0,
             py::arg("threads") = 0)
        .def("hard_observe",
             [](CenteredLdpPsiState& self,
                int source,
                py::object value,
                double response_scale,
                int threads,
                bool clamp) {
                 int sigma = 0;
                 if (self.raw()) {
                     sigma = self.store().encodeToken(
                         source, py::str(value));
                     if (sigma == 0) {
                         throw std::invalid_argument(
                             "unknown hard-observation label for column " +
                             std::to_string(source));
                     }
                 } else {
                     sigma = py::cast<int>(value);
                 }

                 LdpStepSummary summary;
                 {
                     py::gil_scoped_release release;
                     summary = self.hardObserve(
                         source,
                         sigma,
                         response_scale,
                         threads,
                         clamp);
                 }
                 return ldp_step_summary_to_py(summary);
             },
             py::arg("source"),
             py::arg("value"),
             py::arg("response_scale") = 1.0,
             py::arg("threads") = 0,
             py::arg("clamp") = true,
             "Hard-collapse one coordinate to an observed alphabet symbol "
             "and immediately propagate the centered hard-response update. "
             "With clamp=True, keep the observed coordinate fixed during later sweeps.")
        .def("sweep",
             [](CenteredLdpPsiState& self,
                int n,
                const std::string& event,
                uint64_t seed,
                double response_scale,
                int threads,
                bool random_permutation) {
                 LdpStepSummary summary;
                 {
                     py::gil_scoped_release release;
                     summary = self.sweep(
                         n,
                         event,
                         seed,
                         response_scale,
                         threads,
                         random_permutation);
                 }
                 return ldp_step_summary_to_py(summary);
             },
             py::arg("n") = 10,
             py::arg("event") = "mode",
             py::arg("seed") = 1,
             py::arg("response_scale") = 1.0,
             py::arg("threads") = 0,
             py::arg("random_permutation") = false)
        .def("iterate_sweeps",
             [](CenteredLdpPsiState& self,
                int sweeps,
                int n,
                const std::string& event,
                uint64_t seed,
                double response_scale,
                int threads,
                double tol,
                int patience,
                bool random_permutation) {
                 std::vector<LdpStepSummary> history;
                 {
                     py::gil_scoped_release release;
                     history = self.iterateSweeps(
                         sweeps,
                         n,
                         event,
                         seed,
                         response_scale,
                         threads,
                         tol,
                         patience,
                         random_permutation);
                 }

                 py::list out;
                 for (const auto& h : history) {
                     out.append(
                         ldp_step_summary_to_py(h));
                 }
                 return out;
             },
             py::arg("sweeps"),
             py::arg("n") = 10,
             py::arg("event") = "mode",
             py::arg("seed") = 1,
             py::arg("response_scale") = 1.0,
             py::arg("threads") = 0,
             py::arg("tol") = 0.0,
             py::arg("patience") = 1,
             py::arg("random_permutation") = false)

        .def("hard_row",
             [](CenteredLdpPsiState& self) {
                 const auto codes = self.hardCodes();
                 py::list out;
                 for (size_t i = 0; i < codes.size(); ++i) {
                     const int code = codes[i];
                     if (self.raw()) {
                         if (code == 0) {
                             out.append(py::str(""));
                         } else if (self.store().tryGet(
                                        static_cast<int>(i))) {
                             out.append(py::str(
                                 self.store().decodeLabel(
                                     static_cast<int>(i), code)));
                         } else {
                             out.append(py::str(
                                 std::to_string(code)));
                         }
                     } else {
                         out.append(py::int_(code));
                     }
                 }
                 return out;
             })
        .def_property_readonly(
             "kernel_cache_size",
             &CenteredLdpPsiState::kernelCacheSize)
        .def_property_readonly(
             "clamped_count",
             &CenteredLdpPsiState::clampedCount)
        .def("to_python",
             [](CenteredLdpPsiState& self) {
                 return psi_to_python(
                     self.store(),
                     self.psi(),
                     self.raw());
             })
        .def("distribution",
             [](CenteredLdpPsiState& self, int col) {
                 if (col < 0 ||
                     static_cast<size_t>(col) >=
                         self.psi().size()) {
                     throw std::out_of_range(
                         "column out of range");
                 }

                 return distribution_to_python(
                     self.store(),
                     col,
                     self.psi()[
                         static_cast<size_t>(col)],
                     self.raw());
             },
             py::arg("col"))
        .def("dependency_count",
             &CenteredLdpPsiState::dependencyCount,
             py::arg("source"))
        .def_property_readonly(
             "step_count",
             &CenteredLdpPsiState::stepCount)
        .def_property_readonly(
             "size",
             &CenteredLdpPsiState::size);

    m.def("centered_ldp_state_from_psi",
          &centered_ldp_state_from_psi_py,
          py::arg("trees_dir"),
          py::arg("psi"),
          py::arg("raw") = true,
          py::arg("run_dir") = "",
          py::arg("cols_per_shard") = 50000,
          py::arg("tree_ids") = std::vector<int>{},
          "Create a resident centered hard-response LDP state "
          "from a probability-valued Psi.");

    m.def("centered_ldp_state_from_row",
          &centered_ldp_state_from_row_py,
          py::arg("trees_dir"),
          py::arg("row"),
          py::arg("raw") = true,
          py::arg("run_dir") = "",
          py::arg("cols_per_shard") = 50000,
          py::arg("tree_ids") = std::vector<int>{},
          py::arg("threads") = 0,
          "Create a centered hard-response LDP state from a hard row "
          "using Psi_i=phi_i(x_-i).");

    py::class_<LdpPsiState, std::unique_ptr<LdpPsiState>>(m, "LdpPsiState")
        .def("step",
             [](LdpPsiState& self, int target_col, int n,
                const std::string& event, uint64_t seed,
                double response_scale, int threads) {
                 LdpStepSummary summary;
                 {
                     py::gil_scoped_release release;
                     summary = self.step(
                         target_col, n, event, seed,
                         response_scale, threads);
                 }
                 return ldp_step_summary_to_py(summary);
             },
             py::arg("target_col") = -1,
             py::arg("n") = 10,
             py::arg("event") = "mode",
             py::arg("seed") = 1,
             py::arg("response_scale") = 1.0,
             py::arg("threads") = 0)
        .def("iterate",
             [](LdpPsiState& self, int steps, int n,
                const std::string& event, uint64_t seed,
                double response_scale, int threads,
                double tol, int patience) {
                 std::vector<LdpStepSummary> history;
                 {
                     py::gil_scoped_release release;
                     history = self.iterate(
                         steps, n, event, seed, response_scale,
                         threads, tol, patience);
                 }
                 py::list out;
                 for (const auto& h : history) {
                     out.append(ldp_step_summary_to_py(h));
                 }
                 return out;
             },
             py::arg("steps"),
             py::arg("n") = 10,
             py::arg("event") = "mode",
             py::arg("seed") = 1,
             py::arg("response_scale") = 1.0,
             py::arg("threads") = 0,
             py::arg("tol") = 0.0,
             py::arg("patience") = 1)

        .def("sweep",
             [](LdpPsiState& self, int n,
                const std::string& event, uint64_t seed,
                double response_scale, int threads,
                bool random_permutation) {
                 LdpStepSummary summary;
                 {
                     py::gil_scoped_release release;
                     summary = self.sweep(
                         n, event, seed, response_scale,
                         threads, random_permutation);
                 }
                 return ldp_step_summary_to_py(summary);
             },
             py::arg("n") = 10,
             py::arg("event") = "mode",
             py::arg("seed") = 1,
             py::arg("response_scale") = 1.0,
             py::arg("threads") = 0,
             py::arg("random_permutation") = true)
        .def("iterate_sweeps",
             [](LdpPsiState& self, int sweeps, int n,
                const std::string& event, uint64_t seed,
                double response_scale, int threads,
                double tol, int patience,
                bool random_permutation) {
                 std::vector<LdpStepSummary> history;
                 {
                     py::gil_scoped_release release;
                     history = self.iterateSweeps(
                         sweeps, n, event, seed, response_scale,
                         threads, tol, patience, random_permutation);
                 }
                 py::list out;
                 for (const auto& h : history) {
                     out.append(ldp_step_summary_to_py(h));
                 }
                 return out;
             },
             py::arg("sweeps"),
             py::arg("n") = 10,
             py::arg("event") = "mode",
             py::arg("seed") = 1,
             py::arg("response_scale") = 1.0,
             py::arg("threads") = 0,
             py::arg("tol") = 0.0,
             py::arg("patience") = 1,
             py::arg("random_permutation") = true)
        .def("to_python",
             [](LdpPsiState& self) {
                 return psi_to_python(
                     self.store(), self.psi(), self.raw());
             })
        .def("distribution",
             [](LdpPsiState& self, int col) {
                 if (col < 0 ||
                     static_cast<size_t>(col) >= self.psi().size()) {
                     throw std::out_of_range("column out of range");
                 }
                 return distribution_to_python(
                     self.store(), col,
                     self.psi()[static_cast<size_t>(col)],
                     self.raw());
             },
             py::arg("col"))
        .def_property_readonly("step_count", &LdpPsiState::stepCount)
        .def_property_readonly("size", &LdpPsiState::size);

    m.def("ldp_state_from_row",
          &ldp_state_from_row_py,
          py::arg("trees_dir"),
          py::arg("row"),
          py::arg("raw") = true,
          py::arg("run_dir") = "",
          py::arg("cols_per_shard") = 50000,
          py::arg("tree_ids") = std::vector<int>{},
          py::arg("threads") = 0,
          "Create a resident LDP Psi state from a hard row using "
          "Psi_i = phi_i(x_-i) for every learned target.");

    m.def("ldp_state_from_psi",
          &ldp_state_from_psi_py,
          py::arg("trees_dir"),
          py::arg("psi"),
          py::arg("raw") = true,
          py::arg("run_dir") = "",
          py::arg("cols_per_shard") = 50000,
          py::arg("tree_ids") = std::vector<int>{},
          "Create a resident LDP state directly from a saved probability-valued Psi.");


    py::class_<JointParticleState, std::unique_ptr<JointParticleState>>(
            m, "JointParticleState")
        .def("initialize_missing",
             [](JointParticleState& self, int threads) {
                 py::gil_scoped_release release;
                 self.initializeMissing(threads);
             },
             py::arg("threads") = 0)
        .def("step",
             [](JointParticleState& self, int target_col, int threads) {
                 JointStepSummary summary;
                 {
                     py::gil_scoped_release release;
                     summary = self.step(target_col, threads);
                 }
                 return joint_summary_to_py(summary);
             },
             py::arg("target_col") = -1,
             py::arg("threads") = 0,
             "Apply one sequential single-site LSM kernel to every particle. "
             "target_col=-1 uses independent uniform random-scan targets.")
        .def("sweep",
             [](JointParticleState& self, const std::string& mode,
                int updates_per_particle, int threads) {
                 JointStepSummary summary;
                 {
                     py::gil_scoped_release release;
                     summary = self.sweep(mode, updates_per_particle, threads);
                 }
                 return joint_summary_to_py(summary);
             },
             py::arg("mode") = "random_scan",
             py::arg("updates_per_particle") = 0,
             py::arg("threads") = 0,
             "Apply a sequential LSM sweep to every joint-state particle.")
        .def("marginals",
             [](JointParticleState& self) {
                 const ProbabilityRow q = self.marginalsNative();
                 return psi_to_python(self.store(), q, self.raw());
             })
        .def("distribution",
             [](JointParticleState& self, int col) {
                 const ProbabilityRow q = self.marginalsNative();
                 if (col < 0 || static_cast<size_t>(col) >= q.size()) {
                     throw std::out_of_range("column out of range");
                 }
                 return distribution_to_python(
                     self.store(), col, q[static_cast<size_t>(col)], self.raw());
             },
             py::arg("col"))
        .def_property_readonly(
             "n_particles", &JointParticleState::particleCount)
        .def_property_readonly(
             "width", &JointParticleState::width)
        .def_property_readonly(
             "step_count", &JointParticleState::stepCount);

    m.def("joint_state_from_row",
          &joint_state_from_row_py,
          py::arg("trees_dir"),
          py::arg("row"),
          py::arg("n_particles") = 256,
          py::arg("seed") = 1,
          py::arg("raw") = true,
          py::arg("run_dir") = "",
          py::arg("cols_per_shard") = 50000,
          py::arg("tree_ids") = std::vector<int>{},
          py::arg("initialize_missing") = true,
          py::arg("threads") = 0,
          "Create a joint particle representation of an LSM state. Missing "
          "coordinates can be filled by a randomized sequential conditional sweep.");

    m.doc() = "Darkome prediction (single-column and all-columns distributions)";
    m.def("predict_distribution",
          &predict_distribution_py,
          py::arg("trees_dir"),
          py::arg("target_col_id"),
          py::arg("row"),
          py::arg("raw") = false,
          py::arg("run_dir") = "",
          py::arg("cols_per_shard") = 50000);
    m.def("predict_distributions",
          &predict_distributions_py,
          py::arg("trees_dir"),
          py::arg("row"),
          py::arg("raw") = true,
          py::arg("run_dir") = "",
          py::arg("cols_per_shard") = 50000,
          py::arg("tree_ids") = std::vector<int>{});
    m.def("row_to_psi",
          &row_to_psi_py,
          py::arg("trees_dir"),
          py::arg("row"),
          py::arg("raw") = true,
          py::arg("run_dir") = "",
          py::arg("cols_per_shard") = 50000,
          py::arg("tree_ids") = std::vector<int>{},
          "Convert a hard/partial row into probability-valued Psi. Observed "
          "coordinates become deltas; missing coordinates are initialized "
          "synchronously with predict_distribution.");
    m.def("predict_distribution_soft",
          &predict_distribution_soft_py,
          py::arg("trees_dir"),
          py::arg("target_col_id"),
          py::arg("psi"),
          py::arg("raw") = true,
          py::arg("run_dir") = "",
          py::arg("cols_per_shard") = 50000,
          "Evaluate one learned conditional on a probability-valued row.");
    m.def("phi",
          &phi_py,
          py::arg("trees_dir"),
          py::arg("psi"),
          py::arg("raw") = true,
          py::arg("run_dir") = "",
          py::arg("cols_per_shard") = 50000,
          py::arg("tree_ids") = std::vector<int>{},
          py::arg("threads") = 0,
          "Synchronous mean-field LSM update Phi(Psi), parallel over target columns.");

    m.def("empty_tower_check",
          &empty_tower_check_py,
          py::arg("trees_dir"),
          py::arg("top_k") = 20,
          py::arg("raw") = true,
          py::arg("run_dir") = "",
          py::arg("cols_per_shard") = 50000,
          py::arg("tree_ids") = std::vector<int>{},
          py::arg("threads") = 0,
          "Check the empty-state tower identity "
          "sum_sigma p_i(sigma) phi_j(x_i=sigma) = p_j "
          "for every non-trivial learned dependency i->j.");

    m.def("ldp_update",
          &ldp_update_py,
          py::arg("trees_dir"),
          py::arg("psi"),
          py::arg("target_col") = -1,
          py::arg("n") = 10,
          py::arg("event") = "mode",
          py::arg("seed") = 1,
          py::arg("empirical") = py::none(),
          py::arg("response_scale") = 1.0,
          py::arg("raw") = true,
          py::arg("run_dir") = "",
          py::arg("cols_per_shard") = 50000,
          py::arg("tree_ids") = std::vector<int>{},
          py::arg("threads") = 0,
          "One intrinsic LDP event on a probability-valued LSM state. "
          "The selected marginal is replaced by an empirical type; the "
          "cross-variable response is Phi_MF(psi_perturbed)-Phi_MF(psi). "
          "event='mode' gives the highest-probability finite-n empirical type; "
          "event='sample' draws one realization; event='zero_action' is the "
          "exact large-n fixed-point update.");

    m.def("omega",
          &omega_py,
          py::arg("trees_dir"),
          py::arg("row"),
          py::arg("raw") = true,
          py::arg("run_dir") = "",
          py::arg("cols_per_shard") = 50000,
          py::arg("tree_ids") = std::vector<int>{});
    m.def("normalized_profiles_batch", &normalized_profiles_batch_py,
          py::arg("trees_dir"), py::arg("rows"), py::arg("run_dir") = "",
          py::arg("cols_per_shard") = 50000, py::arg("tree_ids") = std::vector<int>{},
          py::arg("threads") = 0, py::arg("eps_floor") = 1e-12);
    m.def("pseudo_code_lengths_batch", &pseudo_code_lengths_batch_py,
          py::arg("trees_dir"), py::arg("rows"), py::arg("run_dir") = "",
          py::arg("cols_per_shard") = 50000, py::arg("tree_ids") = std::vector<int>{},
          py::arg("threads") = 0, py::arg("prob_floor") = 1e-12,
          py::arg("return_per_col") = false,
          "Compute LSM pseudo-description length -sum_i log2 phi_i(x_i|x_-i) per row. "
          "This is a pseudolikelihood score, not a normalized joint code length.");
    m.def("persistence_batch", &persistence_batch_py,
          py::arg("trees_dir"), py::arg("rows"), py::arg("run_dir") = "",
          py::arg("cols_per_shard") = 50000, py::arg("tree_ids") = std::vector<int>{},
          py::arg("threads") = 0, py::arg("prob_floor") = 1e-12,
          "Compute native LSM persistence sum_i log2 phi_i(x_i|x_-i) for a batch of rows.");

    m.def("persistence",
          &persistence_py,
          py::arg("trees_dir"),
          py::arg("row"),
          py::arg("raw") = true,
          py::arg("run_dir") = "",
          py::arg("cols_per_shard") = 50000,
          py::arg("tree_ids") = std::vector<int>{},
          py::arg("return_per_col") = false);
}
#if 0
// Historical alternative implementation retained upstream.
#endif
