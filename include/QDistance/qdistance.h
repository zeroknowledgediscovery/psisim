#pragma once
#include <vector>
#include <map>
#include <utility>
#include <stdexcept>
#include "PredictDistribution/PredictDistribution.h"

/// Compute Jensen–Shannon divergence (in bits) between two discrete pmfs
/// given as label->probability maps. Assumes inputs are normalized (sum=1).
double jensen_shannon_bits(const std::map<int,double>& p,
                           const std::map<int,double>& q);

/// For a single treeId, compute JSD(bits) between two rows' predicted distributions.
double qdistance_for_tree(const PredictDistribution& predictor,
                          int treeId,
                          const std::vector<int>& rowA,
                          const std::vector<int>& rowB);

/// Average JSD(bits) across a set of treeIds (interpreted as columns to average over).
double qdistance(const PredictDistribution& predictor,
                 const std::vector<int>& rowA,
                 const std::vector<int>& rowB,
                 const std::vector<int>& treeIds);

/// Convenience: return per-tree JSDs (bits), useful for diagnostics.
std::vector<std::pair<int,double>> qdistance_per_tree(const PredictDistribution& predictor,
                                                      const std::vector<int>& rowA,
                                                      const std::vector<int>& rowB,
                                                      const std::vector<int>& treeIds);
