// Tree.cpp
#include "Tree/Tree.h"
#include <sstream>
#include <fstream>
#include <vector>

using namespace std;

TreeNode::TreeNode(int id, int obsCount, const map<int, int>& counts) {
    nodeId = id;
    observationCount = obsCount;
    featureCounts = counts;
}
int TreeNode::getId() const {
    return nodeId;
}
int TreeNode::getObservationCount() const {
    return observationCount;
}

const map<int, int>& TreeNode::getFeatureCounts() const { 
    return featureCounts; 
}

LeafNode::LeafNode(int id, int obsCount, const map<int, int>& counts)
    : TreeNode(id, obsCount,counts) {}

bool LeafNode::isLeaf() const {
    return true;
}

json LeafNode::toJSON() const {
    json j;
    j["nodeId"] = nodeId;
    j["count"] = observationCount;
    j["nodeType"] = "leaf";
    j["featureCounts"] = featureCounts;
    return j;
}

SplitNode::SplitNode(
    int id,
    int obsCount,
    int colId,
    int leftCount,
    int rightCount,
    const set<int>& left,
    const set<int>& right,
    const map<int, int>& featureCounts
) : TreeNode(id, obsCount, featureCounts) {
    columnID = colId;
    this->leftCount = leftCount;
    this->rightCount = rightCount;
    leftSubset = left;
    rightSubset = right;
}

bool SplitNode::isLeaf() const {
    return false;
}

void SplitNode::setLeft(unique_ptr<TreeNode> l) {
    left = std::move(l);
}

void SplitNode::setRight(unique_ptr<TreeNode> r) {
    right = std::move(r);
}

TreeNode* SplitNode::getLeft() const {
    return left.get();
}

TreeNode* SplitNode::getRight() const {
    return right.get();
}

bool SplitNode::hasLeft() const {
    return left != nullptr;
}

bool SplitNode::hasRight() const {
    return right != nullptr;
}

int SplitNode::getColumnID() const {
    return columnID;
}

int SplitNode::getLeftCount() const {
    return leftCount;
}

int SplitNode::getRightCount() const {
    return rightCount;
}

const set<int>& SplitNode::getLeftSubset() const {
    return leftSubset;
}

const set<int>& SplitNode::getRightSubset() const {
    return rightSubset;
}

json SplitNode::toJSON() const {
    json j;
    j["nodeId"] = nodeId;
    j["count"] = observationCount;
    j["nodeType"] = "split";
    j["columnID"] = columnID;
    j["leftCount"] = leftCount;
    j["rightCount"] = rightCount;
    j["LSubset"] = leftSubset;
    j["RSubset"] = rightSubset;
    j["LBranch"] = left ? left->toJSON() : nullptr;
    j["RBranch"] = right ? right->toJSON() : nullptr;
    j["featureCounts"] = featureCounts;
    return j;
}




Tree::Tree() : nextNodeId(0) {}

int Tree::generateId() {
    return nextNodeId++;
}

void Tree::createSplit(const SplitInfo& split) {
    int currentNodeId = generateId();
    #ifdef USE_TERMINAL_LOGGING
    cout << "Creating split node with ID: " << currentNodeId << endl;
    #endif
    auto newSplit = make_unique<SplitNode>(
        currentNodeId,
        split.observationCount,
        split.columnID,
        split.leftCount,
        split.rightCount,
        split.leftSubset,
        split.rightSubset,
        split.featureCounts
    );

    SplitNode* newSplitPtr = newSplit.get();

    if (!root) {
        root = std::move(newSplit);
    } else {
        SplitNode* parent = stack.back();
        if (!parent->hasLeft()) {
            parent->setLeft(std::move(newSplit));
        } else if (!parent->hasRight()) {
            parent->setRight(std::move(newSplit));
            stack.pop_back();
        } else {
            throw logic_error("Both children already assigned");
        }
    }

    stack.push_back(newSplitPtr);
}

void Tree::createLeaf(const LeafInfo& leaf) {
    int currentNodeId = generateId();
    #ifdef USE_TERMINAL_LOGGING
    cout << "Creating leaf node with ID: " << currentNodeId << endl;
    #endif
    auto newLeaf = make_unique<LeafNode>(
        currentNodeId, leaf.observationCount, leaf.featureCounts
    );

    if (!root) {
        root = std::move(newLeaf);
        return;
    }

    if (stack.empty()) {
        throw logic_error("No split node to attach leaf to");
    }

    SplitNode* parent = stack.back();
    if (!parent->hasLeft()) {
        parent->setLeft(std::move(newLeaf));
    } else if (!parent->hasRight()) {
        parent->setRight(std::move(newLeaf));
        stack.pop_back();
    } else {
        throw logic_error("Both children already assigned");
    }
}

bool Tree::currentNodeHasLeft() const {
    if (stack.empty()) {
        throw logic_error("No current split node");
    }
    return stack.back()->hasLeft();
}

bool Tree::currentNodeHasRight() const {
    if (stack.empty()) {
        throw logic_error("No current split node");
    }
    return stack.back()->hasRight();
}


int Tree::getCurrentNodeId() const {
    if (stack.empty()) {
        throw logic_error("Tree not started");
    }
    return stack.back()->getId();
}

int Tree::getCurrentColumnId() const {
    if (stack.empty()) throw logic_error("No current split node");
    return stack.back()->getColumnID();
}

bool Tree::isComplete() {
    if (stack.empty()) {
        return true;
    }
    if (stack.size() > 1) {
        return false;
    }
    SplitNode* node = stack.back();
    if (node->hasLeft() && node->hasRight() && node->getId() == 0) {
        return true;
    }
    if (node->hasLeft() && node->hasRight()) {
        stack.pop_back();
        return isComplete();
    }
    return false;
}

// void Tree::writeTreeJSON(ostream& out) const {
//     if (!root) {
//         throw runtime_error("Tree is empty");
//     }
//     out << root->toJSON().dump(2) << endl;
// }

static void collectColumnIds(const TreeNode* node, std::vector<int>& out) {
    if (!node) return;
    if (!node->isLeaf()) {
        const SplitNode* s = static_cast<const SplitNode*>(node);
        out.push_back(s->getColumnID());
        if (s->hasLeft())  collectColumnIds(s->getLeft(),  out);
        if (s->hasRight()) collectColumnIds(s->getRight(), out);
    }
}

void Tree::writeTreeJSON(std::ostream& out, int targetColumn) const {
    if (!root) throw std::runtime_error("Tree is empty");

    json j = root->toJSON();

    std::vector<int> related;
    collectColumnIds(root.get(), related);
    j["relatedNodes"] = related;

    // Root columnID will serve as the file number
    if (!root->isLeaf()) {
        const SplitNode* s = static_cast<const SplitNode*>(root.get());

    }

    // Always append to the log (assume it exists)
    std::ofstream log("node_relations.csv", std::ios::app);
    for (int col : related) {
        log << targetColumn << "," << col << "\n";
    }

    out << j.dump(2) << '\n';
}




set<int> Tree::getCurrentSubset() const {
    if (stack.empty()) {
        throw logic_error("No current split node available.");
    }

    SplitNode* node = stack.back();
    if (!node->hasLeft()) {
        return node->getLeftSubset();
    } else if (!node->hasRight()) {
        return node->getRightSubset();
    } else {
        throw logic_error("Both children already assigned; cannot get current subset.");
    }
}






static string formatSet(const set<int>& s) {
    ostringstream oss;
    oss << "{";
    for (auto it = s.begin(); it != s.end(); ++it) {
        if (it != s.begin()) oss << ",";
        oss << *it;
    }
    oss << "}";
    return oss.str();
}

static void writeNodeDOT(const TreeNode* node, ostream& out) {
    if (node->isLeaf()) {
        const LeafNode* leaf = static_cast<const LeafNode*>(node);
        out << "    node" << leaf->getId()
            << " [label=\"Leaf\\nID=" << leaf->getId()
            << "\\nCount=" << leaf->getObservationCount()
            << "\", shape=box, style=filled, fillcolor=lightgray];\n";
    } else {
        const SplitNode* split = static_cast<const SplitNode*>(node);
        out << "    node" << split->getId()
            << " [label=\"Split\\nID=" << split->getId()
            << "\\nCol=" << split->getColumnID()
            << "\\nCount=" << split->getObservationCount()
            << "\\nLeftCount=" << split->getLeftCount()
            << "\\nRightCount=" << split->getRightCount()
            << "\", shape=ellipse, style=filled, fillcolor=lightblue];\n";

        if (split->hasLeft()) {
            out << "    node" << split->getId()
                << " -> node" << split->getLeft()->getId()
                << " [label=\"" << formatSet(split->getLeftSubset()) << "\"];\n";
            writeNodeDOT(split->getLeft(), out);
        }

        if (split->hasRight()) {
            out << "    node" << split->getId()
                << " -> node" << split->getRight()->getId()
                << " [label=\"" << formatSet(split->getRightSubset()) << "\"];\n";
            writeNodeDOT(split->getRight(), out);
        }
    }
}



void Tree::writeTreeDOT(ostream& out) const {
    if (!root) {
        throw runtime_error("Tree is empty");
    }
    out << "digraph Tree {\n";
    out << "    node [fontname=\"Arial\"];\n";
    writeNodeDOT(root.get(), out);
    out << "}\n";
}

void Tree::writeTreeBinary(ostream& out) const {
    if (!root) throw std::runtime_error("Tree is empty");

    std::function<void(const TreeNode*)> writeNode = [&](const TreeNode* node) {
        if (node->isLeaf()) {
            out.put('L');

            int id = node->getId();
            int count = node->getObservationCount();
            out.write(reinterpret_cast<const char*>(&id), sizeof(int));
            out.write(reinterpret_cast<const char*>(&count), sizeof(int));

            const auto& counts = node->getFeatureCounts();
            int countSize = counts.size();
            out.write(reinterpret_cast<const char*>(&countSize), sizeof(int));
            for (const auto& [key, val] : counts) {
                out.write(reinterpret_cast<const char*>(&key), sizeof(int));
                out.write(reinterpret_cast<const char*>(&val), sizeof(int));
            }
        } else {
            const SplitNode* split = static_cast<const SplitNode*>(node);
            out.put('S');

            int id = split->getId();
            int count = split->getObservationCount();
            int colId = split->getColumnID();
            int leftCount = split->getLeftCount();
            int rightCount = split->getRightCount();
            out.write(reinterpret_cast<const char*>(&id), sizeof(int));
            out.write(reinterpret_cast<const char*>(&count), sizeof(int));
            out.write(reinterpret_cast<const char*>(&colId), sizeof(int));
            out.write(reinterpret_cast<const char*>(&leftCount), sizeof(int));
            out.write(reinterpret_cast<const char*>(&rightCount), sizeof(int));

            auto writeSet = [&](const set<int>& s) {
                int sz = s.size();
                out.write(reinterpret_cast<const char*>(&sz), sizeof(int));
                for (int v : s)
                    out.write(reinterpret_cast<const char*>(&v), sizeof(int));
            };

            writeSet(split->getLeftSubset());
            writeSet(split->getRightSubset());

            const auto& counts = split->getFeatureCounts();
            int countSize = counts.size();
            out.write(reinterpret_cast<const char*>(&countSize), sizeof(int));
            for (const auto& [key, val] : counts) {
                out.write(reinterpret_cast<const char*>(&key), sizeof(int));
                out.write(reinterpret_cast<const char*>(&val), sizeof(int));
            }

            writeNode(split->getLeft());
            writeNode(split->getRight());
        }
    };

    writeNode(root.get());

#ifdef USE_TERMINAL_LOGGING
    cout << "." << flush;
#endif
}


Tree::Tree(std::istream& in) {
    std::function<std::unique_ptr<TreeNode>()> readNode = [&]() -> std::unique_ptr<TreeNode> {
        char type = in.get();
        if (type == 'L') {
            int id, count;
            in.read(reinterpret_cast<char*>(&id), sizeof(int));
            in.read(reinterpret_cast<char*>(&count), sizeof(int));

            int mapSize;
            in.read(reinterpret_cast<char*>(&mapSize), sizeof(int));
            map<int, int> counts;
            for (int i = 0; i < mapSize; ++i) {
                int key, val;
                in.read(reinterpret_cast<char*>(&key), sizeof(int));
                in.read(reinterpret_cast<char*>(&val), sizeof(int));
                counts[key] = val;
            }

            nextNodeId = max(nextNodeId, id + 1);
            return std::make_unique<LeafNode>(id, count, counts);
        } else if (type == 'S') {
            int id, count, columnID, leftCount, rightCount;
            in.read(reinterpret_cast<char*>(&id), sizeof(int));
            in.read(reinterpret_cast<char*>(&count), sizeof(int));
            in.read(reinterpret_cast<char*>(&columnID), sizeof(int));
            in.read(reinterpret_cast<char*>(&leftCount), sizeof(int));
            in.read(reinterpret_cast<char*>(&rightCount), sizeof(int));

            auto readSet = [&]() -> set<int> {
                int sz;
                in.read(reinterpret_cast<char*>(&sz), sizeof(int));
                set<int> s;
                for (int i = 0; i < sz; ++i) {
                    int v;
                    in.read(reinterpret_cast<char*>(&v), sizeof(int));
                    s.insert(v);
                }
                return s;
            };

            set<int> leftSet = readSet();
            set<int> rightSet = readSet();

            int mapSize;
            in.read(reinterpret_cast<char*>(&mapSize), sizeof(int));
            map<int, int> counts;
            for (int i = 0; i < mapSize; ++i) {
                int key, val;
                in.read(reinterpret_cast<char*>(&key), sizeof(int));
                in.read(reinterpret_cast<char*>(&val), sizeof(int));
                counts[key] = val;
            }

            nextNodeId = max(nextNodeId, id + 1);
            auto split = std::make_unique<SplitNode>(id, count, columnID,leftCount,rightCount, leftSet, rightSet, counts);
            split->setLeft(readNode());
            split->setRight(readNode());
            return split;
        } else {
            throw std::runtime_error("Invalid node type in binary stream");
        }
    };

    root = readNode();
}


TreeNode* Tree::getRoot() const {
    return root.get();
}








