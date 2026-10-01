#include "PredictDistribution/PredictDistribution.h"
#include "Tree/Tree.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <mutex>
#include <set>

using namespace std;

namespace {

map<int, double> normalize(const map<int, int>& counts, const TreeNode* node = nullptr) {
    double total = 0.0;
    for (const auto& kv : counts) total += static_cast<double>(kv.second);
    if (total <= 0.0) {
        if (node) {
            throw logic_error(
                "normalize: total count <= 0 at leaf node " + to_string(node->getId()) +
                " observationCount=" + to_string(node->getObservationCount()) +
                " featureCounts.size=" + to_string(counts.size())
            );
        }
        throw logic_error("normalize: total count <= 0");
    }

    map<int, double> probs;
    for (const auto& kv : counts) probs[kv.first] = static_cast<double>(kv.second) / total;
    return probs;
}

map<int, double> mix_probs(const map<int, double>& pLeft,
                           const map<int, double>& pRight,
                           double pi) {
    map<int, double> output = pLeft;
    for (const auto& kv : pRight) if (!output.count(kv.first)) output[kv.first] = 0.0;
    for (auto& kv : output) {
        const double a = pLeft.count(kv.first) ? pLeft.at(kv.first) : 0.0;
        const double b = pRight.count(kv.first) ? pRight.at(kv.first) : 0.0;
        kv.second = pi * a + (1.0 - pi) * b;
    }
    return output;
}

double subtree_mass(TreeNode* node,
                    unordered_map<const TreeNode*, double>& memo) {
    const auto found = memo.find(node);
    if (found != memo.end()) return found->second;

    double mass = 0.0;
    if (node->isLeaf()) {
        for (const auto& kv : node->getFeatureCounts()) mass += static_cast<double>(kv.second);
    } else {
        SplitNode* split = static_cast<SplitNode*>(node);
        if (!split->hasLeft() || !split->hasRight()) throw logic_error("subtree_mass: split is missing a child");
        mass = subtree_mass(split->getLeft(), memo) + subtree_mass(split->getRight(), memo);
    }
    if (mass <= 0.0) throw logic_error("subtree_mass: subtree has zero target mass");
    memo[node] = mass;
    return mass;
}

map<int, double> infer(TreeNode* node,
                       const vector<int>& values,
                       unordered_map<const TreeNode*, double>& massMemo) {
    if (node->isLeaf()) return normalize(node->getFeatureCounts(), node);

    SplitNode* split = static_cast<SplitNode*>(node);
    const int columnID = split->getColumnID();
    if (columnID < 0 || static_cast<size_t>(columnID) >= values.size()) {
        throw out_of_range("infer: column ID out of range: " + to_string(columnID));
    }

    const int v = values[static_cast<size_t>(columnID)];
    if (v != 0 && split->getLeftSubset().count(v)) {
        if (!split->hasLeft()) throw logic_error("infer: missing left child");
        return infer(split->getLeft(), values, massMemo);
    }
    if (v != 0 && split->getRightSubset().count(v)) {
        if (!split->hasRight()) throw logic_error("infer: missing right child");
        return infer(split->getRight(), values, massMemo);
    }

    if (!split->hasLeft() || !split->hasRight()) throw logic_error("infer: missing child at split during marginalization");

    const double nLeft = subtree_mass(split->getLeft(), massMemo);
    const double nRight = subtree_mass(split->getRight(), massMemo);
    const double denominator = nLeft + nRight;
    if (denominator <= 0.0) throw logic_error("infer: constructed child mass <= 0");
    const double pi = nLeft / denominator;

    const map<int, double> pLeft = infer(split->getLeft(), values, massMemo);
    const map<int, double> pRight = infer(split->getRight(), values, massMemo);
    return mix_probs(pLeft, pRight, pi);
}




void add_scaled(ProbabilityDistribution& dst,
                const ProbabilityDistribution& src,
                double weight) {
    if (weight == 0.0) return;
    for (const auto& kv : src) dst[kv.first] += weight * kv.second;
}

void normalize_in_place(ProbabilityDistribution& q, double total) {
    if (!(total > 0.0)) {
        throw invalid_argument("predictSoft: cannot normalize zero probability mass");
    }
    for (auto& kv : q) kv.second /= total;
}

ProbabilityDistribution infer_soft(
        TreeNode* node,
        const ProbabilityRow& psi,
        vector<ProbabilityDistribution>& overrides,
        vector<unsigned char>& hasOverride,
        unordered_map<const TreeNode*, double>& massMemo) {
    if (node->isLeaf()) return normalize(node->getFeatureCounts(), node);

    SplitNode* split = static_cast<SplitNode*>(node);
    const int columnID = split->getColumnID();
    if (columnID < 0 || static_cast<size_t>(columnID) >= psi.size()) {
        throw out_of_range(
            "predictSoft: column ID out of range: " + to_string(columnID));
    }
    if (!split->hasLeft() || !split->hasRight()) {
        throw logic_error("predictSoft: split is missing a child");
    }

    const size_t ci = static_cast<size_t>(columnID);
    const ProbabilityDistribution& q =
        hasOverride[ci] ? overrides[ci] : psi[ci];

    if (q.empty()) {
        throw invalid_argument(
            "predictSoft: every Psi coordinate must be a non-empty distribution");
    }

    // For a probability-valued predictor, route each category's mass exactly
    // once. Categories explicitly assigned to a child take that branch with
    // probability one. Categories unresolved by this split follow the same
    // learned subtree-mass marginalization used by hard missing-value
    // inference. The resulting child-specific q_j is the conditional
    // distribution of this variable given the branch. This remains exact if
    // the same predictor variable appears again deeper in the tree.
    double unresolvedMass = 0.0;
    for (const auto& kv : q) {
        const double p = kv.second;
        if (!std::isfinite(p) || p < 0.0) {
            throw invalid_argument(
                "predictSoft: probabilities must be finite and non-negative");
        }
        if (p == 0.0) continue;
        const int code = kv.first;
        if (!split->getLeftSubset().count(code) &&
            !split->getRightSubset().count(code)) {
            unresolvedMass += p;
        }
    }

    double pi = 0.0;
    if (unresolvedMass > 0.0) {
        const double nLeft = subtree_mass(split->getLeft(), massMemo);
        const double nRight = subtree_mass(split->getRight(), massMemo);
        const double denom = nLeft + nRight;
        if (!(denom > 0.0)) {
            throw logic_error("predictSoft: child mass is zero");
        }
        pi = nLeft / denom;
    }

    ProbabilityDistribution leftQ;
    ProbabilityDistribution rightQ;
    double leftMass = 0.0;
    double rightMass = 0.0;

    for (const auto& kv : q) {
        const int code = kv.first;
        const double p = kv.second;
        if (!(p > 0.0)) continue;

        if (split->getLeftSubset().count(code)) {
            leftQ[code] += p;
            leftMass += p;
        } else if (split->getRightSubset().count(code)) {
            rightQ[code] += p;
            rightMass += p;
        } else {
            const double lp = p * pi;
            const double rp = p * (1.0 - pi);
            if (lp > 0.0) {
                leftQ[code] += lp;
                leftMass += lp;
            }
            if (rp > 0.0) {
                rightQ[code] += rp;
                rightMass += rp;
            }
        }
    }

    const double branchTotal = leftMass + rightMass;
    if (!(branchTotal > 0.0)) {
        throw logic_error("predictSoft: split received zero input probability mass");
    }

    // q should already be normalized. Renormalize branch weights against the
    // accumulated total to remain robust to harmless floating-point drift.
    leftMass /= branchTotal;
    rightMass /= branchTotal;

    ProbabilityDistribution result;

    const bool hadOld = hasOverride[ci] != 0;
    ProbabilityDistribution old;
    if (hadOld) old = overrides[ci];

    if (leftMass > 0.0) {
        normalize_in_place(leftQ, leftMass * branchTotal);
        overrides[ci] = std::move(leftQ);
        hasOverride[ci] = 1;
        const ProbabilityDistribution pLeft =
            infer_soft(split->getLeft(), psi, overrides, hasOverride, massMemo);
        add_scaled(result, pLeft, leftMass);
    }

    if (rightMass > 0.0) {
        normalize_in_place(rightQ, rightMass * branchTotal);
        overrides[ci] = std::move(rightQ);
        hasOverride[ci] = 1;
        const ProbabilityDistribution pRight =
            infer_soft(split->getRight(), psi, overrides, hasOverride, massMemo);
        add_scaled(result, pRight, rightMass);
    }

    if (hadOld) {
        overrides[ci] = std::move(old);
        hasOverride[ci] = 1;
    } else {
        overrides[ci].clear();
        hasOverride[ci] = 0;
    }

    double total = 0.0;
    for (const auto& kv : result) total += kv.second;
    if (!(total > 0.0)) {
        throw logic_error("predictSoft: empty output distribution");
    }
    for (auto& kv : result) kv.second /= total;
    return result;
}


void collect_used_columns(const TreeNode* node, set<int>& out) {
    if (!node || node->isLeaf()) return;
    const SplitNode* split = static_cast<const SplitNode*>(node);
    out.insert(split->getColumnID());
    if (split->hasLeft()) collect_used_columns(split->getLeft(), out);
    if (split->hasRight()) collect_used_columns(split->getRight(), out);
}


map<int, int> probs_to_pseudocounts(const map<int, double>& probs, int scale = 1000000) {
    map<int, int> output;
    for (const auto& kv : probs) output[kv.first] = static_cast<int>(std::llround(kv.second * static_cast<double>(scale)));
    long long sum = 0;
    for (const auto& kv : output) sum += kv.second;
    if (sum == 0) throw logic_error("probs_to_pseudocounts: rounded to zero");
    return output;
}

}

PredictDistribution::PredictDistribution(const std::string& dirPath)
    : directory(dirPath) {}

shared_ptr<Tree> PredictDistribution::getTree(int treeId) const {
    {
        shared_lock<shared_mutex> lock(treeCacheMutex);
        auto it = treeCache.find(treeId);
        if (it != treeCache.end()) return it->second;
    }

    // Deserialize outside the exclusive cache lock. Different tree IDs are
    // independent, so first-time loads can proceed concurrently. A second
    // thread may race on the same ID; the insertion-time recheck below keeps
    // only one cached instance.
    ostringstream filename;
    filename << directory << "/tree_" << treeId << ".bin";
    ifstream inFile(filename.str(), ios::binary);
    if (!inFile) throw runtime_error("Could not open tree file: " + filename.str());

    auto loaded = make_shared<Tree>(inFile);
    if (!loaded->getRoot()) throw runtime_error("Tree has no root");

    unique_lock<shared_mutex> lock(treeCacheMutex);
    auto it = treeCache.find(treeId);
    if (it != treeCache.end()) return it->second;
    treeCache.emplace(treeId, loaded);
    return loaded;
}

void PredictDistribution::preload(const vector<int>& treeIds) const {
    for (int treeId : treeIds) (void)getTree(treeId);
}

ProbabilityDistribution PredictDistribution::predictProbability(
        int treeId, const vector<int>& values) const {
    const auto tree = getTree(treeId);
    TreeNode* root = tree->getRoot();
    if (!root) throw runtime_error("Tree has no root");

    unordered_map<const TreeNode*, double> massMemo;
    return infer(root, values, massMemo);
}


set<int> PredictDistribution::usedColumns(int treeId) const {
    const auto tree = getTree(treeId);
    set<int> out;
    collect_used_columns(tree->getRoot(), out);
    return out;
}

map<int, int> PredictDistribution::predict(
        int treeId, const vector<int>& values) const {
    return probs_to_pseudocounts(predictProbability(treeId, values));
}


ProbabilityDistribution PredictDistribution::predictSoft(
        int treeId, const ProbabilityRow& psi) const {
    const auto tree = getTree(treeId);
    TreeNode* root = tree->getRoot();
    if (!root) throw runtime_error("Tree has no root");

    vector<ProbabilityDistribution> overrides(psi.size());
    vector<unsigned char> hasOverride(psi.size(), 0);
    unordered_map<const TreeNode*, double> massMemo;
    return infer_soft(root, psi, overrides, hasOverride, massMemo);
}
