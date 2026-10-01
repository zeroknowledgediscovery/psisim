#include "QDistance/qdistance.h"
#include <cmath>
#include <set>
#include <numeric>
#include <limits>   
#include <iostream>

// ---------- helpers ----------

static std::map<int,double> normalize_counts_to_probs(const std::map<int,int>& counts) {
    long long total = 0;
    for (const auto& kv : counts) total += kv.second;
    if (total <= 0) throw std::logic_error("normalize_counts_to_probs: total count <= 0");

    std::map<int,double> probs;                // std::map has no reserve() dammit
    for (const auto& kv : counts) {
        probs.emplace(kv.first, static_cast<double>(kv.second) / static_cast<double>(total));
    }
    return probs;
}


static inline double safe_log2(double x) {
    // handle x==0 by convention 0*log(0)=0 in KL/JSD summations
    return std::log(x) / std::log(2.0);
}

static double kl_bits(const std::map<int,double>& p, const std::map<int,double>& m) {
    // sum_i p_i * log2(p_i / m_i), with 0 * log(0/.) = 0 by ""convention""
    double s = 0.0;
    auto itP = p.begin();
    auto itM = m.begin();
    // Since p and m may have different supports, just iterate p and find m_i
    for (const auto& kv : p) {
        const int   label = kv.first;
        const double pi   = kv.second;
        if (pi <= 0.0) continue;
        auto mit = m.find(label);
        if (mit == m.end() || mit->second <= 0.0) {
            // If m_i == 0 but p_i > 0, KL is infinite; practically, this should not happen
            // for m = (p+q)/2 unless both p_i and q_i are 0 (then p_i==0), so guard anyway:
            return std::numeric_limits<double>::infinity();
        }
        s += pi * (safe_log2(pi) - safe_log2(mit->second));
    }
    return s;
}





// ---------- public API ----------

double jensen_shannon_bits(const std::map<int,double>& p,
                           const std::map<int,double>& q) {
    // Build union of labels and materialize pmfs over that union (missing → 0)
    std::map<int,double> P, Q, M;
    P = p;
    Q = q;

    // Compute M = (P+Q)/2 over the union of supports
    // First ensure keys exist in both maps (with 0s), so m[label] always defined.
    for (const auto& kv : P) if (!Q.count(kv.first)) Q.emplace(kv.first, 0.0);
    for (const auto& kv : Q) if (!P.count(kv.first)) P.emplace(kv.first, 0.0);

    for (const auto& kv : P) {
        const int label = kv.first;
        const double pi = kv.second;
        const double qi = Q[label];
        M[label] = 0.5 * (pi + qi);
    }

    // KL(P||M) and KL(Q||M)
    const double dPM = kl_bits(P, M);
    const double dQM = kl_bits(Q, M);
    if (!std::isfinite(dPM) || !std::isfinite(dQM)) {
        // Extremely corner cases should not happen with valid M; protect anyway
        return std::numeric_limits<double>::infinity();
    }

    //std::cout << 0.5 * (dPM + dQM) << std::endl;
    
    return sqrt(0.5 * (dPM + dQM));
}

double qdistance_for_tree(const PredictDistribution& predictor,
                          int treeId,
                          const std::vector<int>& rowA,
                          const std::vector<int>& rowB) {
    // Get counts for each row from the existing predictor
    const std::map<int,int> cntA = predictor.predict(treeId, rowA);
    const std::map<int,int> cntB = predictor.predict(treeId, rowB);

    // Normalize to pmfs
    const std::map<int,double> pA = normalize_counts_to_probs(cntA);
    const std::map<int,double> pB = normalize_counts_to_probs(cntB);

    // JSD in bits
    return jensen_shannon_bits(pA, pB);
}

double qdistance(const PredictDistribution& predictor,
                 const std::vector<int>& rowA,
                 const std::vector<int>& rowB,
                 const std::vector<int>& treeIds) {
    if (treeIds.empty())
        throw std::invalid_argument("qdistance: treeIds is empty");

    double sum = 0.0;
    std::size_t used = 0;
    for (int tid : treeIds) {
        const double jsd = qdistance_for_tree(predictor, tid, rowA, rowB);
        if (std::isfinite(jsd)) { sum += jsd; ++used; }
        // If a tree returns an infinite JSD due to degenerate M (should not), we skip it.
    }
    if (used == 0) throw std::runtime_error("qdistance: no usable trees");
    return sum / static_cast<double>(used);
}

std::vector<std::pair<int,double>> qdistance_per_tree(const PredictDistribution& predictor,
                                                      const std::vector<int>& rowA,
                                                      const std::vector<int>& rowB,
                                                      const std::vector<int>& treeIds) {
    std::vector<std::pair<int,double>> out;
    out.reserve(treeIds.size());
    for (int tid : treeIds) {
        out.emplace_back(tid, qdistance_for_tree(predictor, tid, rowA, rowB));
    }
    return out;
}
