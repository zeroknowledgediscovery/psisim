// PsiSim-owned native dynamics.
//
// This file is NOT part of the vendored LSM runtime and is never touched by
// scripts/sync_lsm_runtime.sh. It only uses the public PredictDistribution
// and SourceMapStore APIs.
//
// 1. used_columns(trees_dir, tree_ids)
//      For every learned target j, the sorted source columns i that the
//      native tree for j splits on (the learned dependency graph).
//
// 2. PropagationPsiState: propagation-only centered dynamics.
//
//    The required invariant is: no external intervention => no motion.
//    The state only moves when a hard observation X_i = sigma is imposed.
//    The observation creates the wave-0 perturbation
//
//        Delta_i^(0) = delta_sigma - p_i^before,
//
//    and every dependent, non-clamped target j receives the centered
//    hard-response
//
//        r_{j<-i} = sum_s Delta_i(s) K_{j<-i}(.|s),
//        K_{j<-i}(.|s) = phi_j(x_empty with X_i = s),
//        p_j^+ = Proj_simplex(p_j + r_{j<-i})        (response scale 1).
//
//    The actual induced change Delta_j = p_j^+ - p_j (after projection)
//    becomes the perturbation that j propagates in the next wave. A wave
//    propagates ONLY the deltas newly induced by the preceding wave, never
//    the accumulated change from the original state, so feedback loops do
//    not double-count. When several sources of one wave hit the same target,
//    their responses are summed and projected once. Clamped coordinates are
//    never modified as targets.
//
//    The kernel, response accumulation and simplex projection reproduce the
//    vendored CenteredLdpPsiState operator exactly (wave 0 equals
//    CenteredLdpPsiState.hard_observe; see webapp/tests).

#include "PredictDistribution/PredictDistribution.h"
#include "SourceMaps/SourceMaps.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <set>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
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

// ---------------------------------------------------------------------------
// Operator pieces. These mirror bindings/predict_distribution_py.cpp
// (simplex_project, CenteredHardKernelCache, centered_hard_response_delta,
// apply_centered_response, distribution_difference, tv_distance) so that the
// same centered response is applied. Kept identical on purpose; the
// regression test compares against the vendored operator.
// ---------------------------------------------------------------------------

ProbabilityDistribution simplex_project(
        const ProbabilityDistribution& q,
        bool* changed) {
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
        const double theta = (cssv - 1.0) / static_cast<double>(j + 1);
        if (u[j] - theta > 0.0) rho = static_cast<int>(j);
    }
    if (rho < 0) throw std::logic_error("simplex projection failed");
    cssv = 0.0;
    for (int j = 0; j <= rho; ++j) cssv += u[static_cast<size_t>(j)];
    const double theta = (cssv - 1.0) / static_cast<double>(rho + 1);

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

struct KernelKey {
    int source;
    int target;
    int sigma;
    bool operator==(const KernelKey& o) const {
        return source == o.source && target == o.target && sigma == o.sigma;
    }
};

struct KernelKeyHash {
    size_t operator()(const KernelKey& k) const noexcept {
        size_t h = static_cast<size_t>(k.source + 0x9e3779b9);
        h ^= static_cast<size_t>(k.target + 0x85ebca6b) + (h << 6) + (h >> 2);
        h ^= static_cast<size_t>(k.sigma + 0xc2b2ae35) + (h << 6) + (h >> 2);
        return h;
    }
};

class KernelCache {
public:
    // K_{target<-source}(.|sigma) = phi_target(x_empty with X_source = sigma)
    ProbabilityDistribution get(PredictDistribution& predictor,
                                int source, int target, int sigma,
                                size_t width) {
        const KernelKey key{source, target, sigma};
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
    std::unordered_map<KernelKey, ProbabilityDistribution, KernelKeyHash> cache_;
};

ProbabilityDistribution centered_response(
        PredictDistribution& predictor, KernelCache& cache,
        int source, int target,
        const ProbabilityDistribution& delta, size_t width) {
    ProbabilityDistribution response;
    for (const auto& kv : delta) {
        const int sigma = kv.first;
        const double weight = kv.second;
        if (std::abs(weight) <= 1e-18) continue;
        const ProbabilityDistribution q =
            cache.get(predictor, source, target, sigma, width);
        for (const auto& out : q) response[out.first] += weight * out.second;
    }
    return response;
}

ProbabilityDistribution apply_response(
        const ProbabilityDistribution& current,
        const ProbabilityDistribution& response,
        double response_scale,
        bool* projected) {
    std::set<int> keys;
    for (const auto& kv : current) keys.insert(kv.first);
    for (const auto& kv : response) keys.insert(kv.first);
    ProbabilityDistribution candidate;
    for (int code : keys) {
        const double cur = current.count(code) ? current.at(code) : 0.0;
        const double dr = response.count(code) ? response.at(code) : 0.0;
        candidate[code] = cur + response_scale * dr;
    }
    return simplex_project(candidate, projected);
}

ProbabilityDistribution difference(const ProbabilityDistribution& a,
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

double half_l1(const ProbabilityDistribution& d) {
    double s = 0.0;
    for (const auto& kv : d) s += std::abs(kv.second);
    return 0.5 * s;
}

double tv_distance(const ProbabilityDistribution& a,
                   const ProbabilityDistribution& b) {
    return half_l1(difference(a, b));
}

// ---------------------------------------------------------------------------
// Propagation-only state
// ---------------------------------------------------------------------------

struct WaveSummary {
    int wave = 0;                 // 0 = hard observation, 1.. = later waves
    int sources = 0;              // coordinates whose delta was propagated
    int targets = 0;              // non-clamped targets that received a response
    int changed = 0;              // coordinates whose distribution moved
    int projected_count = 0;
    double mean_tv = 0.0;         // over learned coordinates
    double max_tv = 0.0;
    int max_col = -1;
    double source_delta_tv = 0.0; // sum of TV norms of propagated deltas
    int pending = 0;              // coordinates carrying a delta for next wave
    double pending_tv = 0.0;      // sum of TV norms of those deltas
};

py::dict summary_to_py(const WaveSummary& s) {
    py::dict d;
    d["wave"] = s.wave;
    d["sources"] = s.sources;
    d["targets"] = s.targets;
    d["changed"] = s.changed;
    d["projected_count"] = s.projected_count;
    d["mean_tv"] = s.mean_tv;
    d["max_tv"] = s.max_tv;
    d["max_col"] = s.max_col;
    d["source_delta_tv"] = s.source_delta_tv;
    d["pending"] = s.pending;
    d["pending_tv"] = s.pending_tv;
    return d;
}

class PropagationPsiState {
public:
    PropagationPsiState(const std::string& trees_dir,
                        const std::string& run_dir,
                        py::object psi_obj,
                        int cols_per_shard,
                        std::vector<int> tree_ids)
        : treesDir_(fs::absolute(trees_dir)),
          runDir_(run_dir.empty()
                      ? sourcemaps::SourceMapStore::inferRunDir(treesDir_)
                      : fs::absolute(run_dir)),
          store_(runDir_, cols_per_shard),
          treeIds_(std::move(tree_ids)),
          predictor_(std::make_shared<PredictDistribution>(treesDir_.string())) {
        if (!fs::exists(runDir_ / "source_maps")) {
            throw std::runtime_error("run_dir must contain source_maps/");
        }
        if (treeIds_.empty()) treeIds_ = discover_tree_ids(treesDir_);
        if (treeIds_.empty()) throw std::runtime_error("No tree_*.bin found");

        psi_ = python_to_psi(psi_obj);
        learned_.insert(treeIds_.begin(), treeIds_.end());

        predictor_->preload(treeIds_);
        const size_t width = psi_.size();
        targets_.assign(width, {});
        for (int target : treeIds_) {
            for (int source : predictor_->usedColumns(target)) {
                if (source == target || source < 0 ||
                    static_cast<size_t>(source) >= width ||
                    !learned_.count(source)) {
                    continue;
                }
                targets_[static_cast<size_t>(source)].push_back(target);
            }
        }
        for (auto& t : targets_) {
            std::sort(t.begin(), t.end());
            t.erase(std::unique(t.begin(), t.end()), t.end());
        }
    }

    // Wave 0: impose X_source = value and propagate Delta^(0) once.
    WaveSummary hardObserve(int source, const std::string& value,
                            bool clamp, int threads) {
        checkColumn(source);
        if (!learned_.count(source)) {
            throw std::invalid_argument("source column has no learned tree");
        }
        if (clamped_.count(source)) {
            throw std::invalid_argument(
                "column " + std::to_string(source) + " is already clamped");
        }
        const int sigma = store_.encodeToken(source, value);
        if (sigma == 0) {
            throw std::invalid_argument(
                "unknown hard-observation label for column " +
                std::to_string(source));
        }
        ProbabilityDistribution point;
        point[sigma] = 1.0;

        std::map<int, ProbabilityDistribution> wave_sources;
        const ProbabilityDistribution delta0 =
            difference(point, psi_[static_cast<size_t>(source)]);
        if (!delta0.empty()) wave_sources[source] = delta0;

        ProbabilityRow next = psi_;
        next[static_cast<size_t>(source)] = point;

        wave_ = 0;
        WaveSummary out;
        {
            py::gil_scoped_release release;
            out = propagate(wave_sources, next, threads,
                            /*exclude_from_pending=*/source);
        }
        if (clamp) clamped_[source] = sigma;
        return out;
    }

    // One later wave: propagate exactly the deltas induced by the previous
    // wave. With nothing pending this is the identity, bit for bit.
    WaveSummary wave(int threads) {
        std::map<int, ProbabilityDistribution> wave_sources;
        wave_sources.swap(pending_);
        ++wave_;
        ProbabilityRow next = psi_;
        py::gil_scoped_release release;
        return propagate(wave_sources, next, threads, /*exclude=*/-1);
    }

    void discardPending() { pending_.clear(); }

    py::list toPython() {
        py::list out(psi_.size());
        for (size_t i = 0; i < psi_.size(); ++i) {
            out[static_cast<ssize_t>(i)] =
                distributionToPython(static_cast<int>(i), psi_[i]);
        }
        return out;
    }

    py::dict distribution(int col) {
        checkColumn(col);
        return distributionToPython(col, psi_[static_cast<size_t>(col)]);
    }

    py::dict pending() {
        py::dict out;
        for (const auto& kv : pending_) {
            out[py::int_(kv.first)] = distributionToPython(kv.first, kv.second);
        }
        return out;
    }

    py::dict clamped() {
        py::dict out;
        for (const auto& kv : clamped_) {
            out[py::int_(kv.first)] =
                py::str(store_.decodeLabel(kv.first, kv.second));
        }
        return out;
    }

    std::vector<int> dependencyTargets(int source) {
        checkColumn(source);
        return targets_[static_cast<size_t>(source)];
    }

    size_t size() const { return psi_.size(); }
    int waveIndex() const { return wave_; }
    size_t pendingCount() const { return pending_.size(); }
    size_t clampedCount() const { return clamped_.size(); }
    size_t kernelCacheSize() const { return cache_.size(); }

private:
    void checkColumn(int col) const {
        if (col < 0 || static_cast<size_t>(col) >= psi_.size()) {
            throw std::out_of_range("column out of range");
        }
    }

    ProbabilityRow python_to_psi(py::object psi_obj) {
        py::sequence seq = py::reinterpret_borrow<py::sequence>(psi_obj);
        const ssize_t n = py::len(seq);
        if (n <= 0) throw std::invalid_argument("Psi must be non-empty");
        ProbabilityRow out(static_cast<size_t>(n));
        for (ssize_t i = 0; i < n; ++i) {
            py::handle obj = seq[i];
            if (!py::isinstance<py::dict>(obj)) {
                throw std::invalid_argument("each Psi coordinate must be a dict");
            }
            py::dict d = py::reinterpret_borrow<py::dict>(obj);
            ProbabilityDistribution q;
            double total = 0.0;
            for (auto item : d) {
                const double p = py::cast<double>(item.second);
                if (!std::isfinite(p) || p < 0.0) {
                    throw std::invalid_argument(
                        "Psi probabilities must be finite and non-negative");
                }
                const int code = store_.encodeToken(
                    static_cast<int>(i), py::str(item.first));
                if (code == 0) {
                    throw std::invalid_argument(
                        "Psi contains an unknown label in column " +
                        std::to_string(i));
                }
                q[code] += p;
                total += p;
            }
            if (!(total > 0.0)) {
                throw std::invalid_argument(
                    "every Psi coordinate must have positive total mass");
            }
            for (auto& kv : q) kv.second /= total;
            out[static_cast<size_t>(i)] = std::move(q);
        }
        return out;
    }

    py::dict distributionToPython(int col, const ProbabilityDistribution& q) {
        py::dict out;
        for (const auto& kv : q) {
            out[py::str(store_.decodeLabel(col, kv.first))] = kv.second;
        }
        return out;
    }

    // Apply one wave of centered responses from `sources` on top of `next`
    // (which already holds any source replacement), then record the actual
    // induced changes as the pending deltas of the next wave.
    WaveSummary propagate(const std::map<int, ProbabilityDistribution>& sources,
                          ProbabilityRow& next, int threads, int exclude) {
        const size_t width = psi_.size();

        // target -> contributing sources (ascending, deterministic order)
        std::map<int, std::vector<int>> incoming;
        for (const auto& kv : sources) {
            for (int target : targets_[static_cast<size_t>(kv.first)]) {
                if (clamped_.count(target)) continue;
                incoming[target].push_back(kv.first);
            }
        }
        std::vector<std::pair<int, std::vector<int>>> jobs(
            incoming.begin(), incoming.end());

        const int nt = threads > 0 ? threads : omp_get_max_threads();
        std::atomic<int> projected_count{0};
        std::atomic<bool> failed{false};
        std::exception_ptr first_error;
        std::mutex error_mutex;

        #pragma omp parallel for schedule(dynamic) num_threads(nt)
        for (ssize_t k = 0; k < static_cast<ssize_t>(jobs.size()); ++k) {
            if (failed.load(std::memory_order_relaxed)) continue;
            try {
                const int target = jobs[static_cast<size_t>(k)].first;
                const auto& from = jobs[static_cast<size_t>(k)].second;
                ProbabilityDistribution total;
                for (int source : from) {
                    const ProbabilityDistribution r = centered_response(
                        *predictor_, cache_, source, target,
                        sources.at(source), width);
                    if (from.size() == 1) {
                        total = r;
                    } else {
                        for (const auto& kv : r) total[kv.first] += kv.second;
                    }
                }
                bool projected = false;
                next[static_cast<size_t>(target)] = apply_response(
                    psi_[static_cast<size_t>(target)], total, 1.0, &projected);
                if (projected) {
                    projected_count.fetch_add(1, std::memory_order_relaxed);
                }
            } catch (...) {
                failed.store(true, std::memory_order_relaxed);
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!first_error) first_error = std::current_exception();
            }
        }
        if (first_error) std::rethrow_exception(first_error);

        WaveSummary out;
        out.wave = wave_;
        out.sources = static_cast<int>(sources.size());
        out.targets = static_cast<int>(jobs.size());
        out.projected_count = projected_count.load();
        for (const auto& kv : sources) out.source_delta_tv += half_l1(kv.second);

        // Newly induced deltas. Any delta still pending (e.g. a propagation
        // stopped early before another answer) is merged so that every
        // induced change is propagated exactly once.
        double sum_tv = 0.0;
        for (size_t i = 0; i < width; ++i) {
            const ProbabilityDistribution d = difference(next[i], psi_[i]);
            if (d.empty()) continue;
            ++out.changed;
            const double tv = half_l1(d);
            if (learned_.count(static_cast<int>(i))) {
                sum_tv += tv;
                if (tv > out.max_tv) {
                    out.max_tv = tv;
                    out.max_col = static_cast<int>(i);
                }
            }
            if (static_cast<int>(i) == exclude) continue;
            auto& slot = pending_[static_cast<int>(i)];
            for (const auto& kv : d) slot[kv.first] += kv.second;
        }
        for (auto it = pending_.begin(); it != pending_.end();) {
            for (auto jt = it->second.begin(); jt != it->second.end();) {
                if (std::abs(jt->second) <= 1e-18) jt = it->second.erase(jt);
                else ++jt;
            }
            if (it->second.empty()) it = pending_.erase(it);
            else ++it;
        }
        out.mean_tv = sum_tv / static_cast<double>(treeIds_.size());
        out.pending = static_cast<int>(pending_.size());
        for (const auto& kv : pending_) out.pending_tv += half_l1(kv.second);

        psi_.swap(next);
        return out;
    }

    fs::path treesDir_;
    fs::path runDir_;
    sourcemaps::SourceMapStore store_;
    std::vector<int> treeIds_;
    std::unordered_set<int> learned_;
    std::shared_ptr<PredictDistribution> predictor_;
    KernelCache cache_;
    ProbabilityRow psi_;
    std::vector<std::vector<int>> targets_;
    std::map<int, ProbabilityDistribution> pending_;
    std::unordered_map<int, int> clamped_;
    int wave_ = 0;
};

}  // namespace

PYBIND11_MODULE(psisim_dynamics, m) {
    m.doc() = "PsiSim-owned native dynamics: dependency graph and "
              "propagation-only centered waves";

    m.def("used_columns", &used_columns_py,
          py::arg("trees_dir"),
          py::arg("tree_ids") = std::vector<int>{},
          "Return {target: sorted source columns used by tree_target}.");

    py::class_<PropagationPsiState, std::unique_ptr<PropagationPsiState>>(
            m, "PropagationPsiState")
        .def(py::init<const std::string&, const std::string&, py::object,
                      int, std::vector<int>>(),
             py::arg("trees_dir"),
             py::arg("run_dir"),
             py::arg("psi"),
             py::arg("cols_per_shard") = 50000,
             py::arg("tree_ids") = std::vector<int>{},
             "Resident propagation-only state initialised at a raw-label Psi.")
        .def("hard_observe",
             [](PropagationPsiState& self, int source, py::object value,
                bool clamp, int threads) {
                 return summary_to_py(self.hardObserve(
                     source, py::str(value), clamp, threads));
             },
             py::arg("source"), py::arg("value"),
             py::arg("clamp") = true, py::arg("threads") = 0,
             "Impose X_source=value (wave 0): collapse p_source to a point "
             "mass and propagate Delta=delta_sigma-p_source once.")
        .def("wave",
             [](PropagationPsiState& self, int threads) {
                 return summary_to_py(self.wave(threads));
             },
             py::arg("threads") = 0,
             "Propagate exactly the deltas induced by the previous wave. "
             "The identity when nothing is pending.")
        .def("discard_pending", &PropagationPsiState::discardPending)
        .def("to_python", &PropagationPsiState::toPython)
        .def("distribution", &PropagationPsiState::distribution, py::arg("col"))
        .def("pending", &PropagationPsiState::pending,
             "{column: {label: delta}} to be propagated by the next wave.")
        .def("clamped", &PropagationPsiState::clamped)
        .def("dependency_targets", &PropagationPsiState::dependencyTargets,
             py::arg("source"))
        .def_property_readonly("size", &PropagationPsiState::size)
        .def_property_readonly("wave_index", &PropagationPsiState::waveIndex)
        .def_property_readonly("pending_count", &PropagationPsiState::pendingCount)
        .def_property_readonly("clamped_count", &PropagationPsiState::clampedCount)
        .def_property_readonly("kernel_cache_size",
                               &PropagationPsiState::kernelCacheSize);
}
