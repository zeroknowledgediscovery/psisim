// Tree.h
#ifndef TREE_H
#define TREE_H

#include <memory>
#include <vector>
#include <map>
#include <set>
#include <iostream>
#include <nlohmann/json.hpp>

using namespace std;
using json = nlohmann::json;

struct SplitInfo {
    int observationCount;
    int columnID;
    int leftCount;
    int rightCount;
    set<int> leftSubset;
    set<int> rightSubset;
    map<int, int> featureCounts;
};

struct LeafInfo {
    int observationCount;
    map<int, int> featureCounts;
};

class TreeNode {
protected:
    int nodeId;
    int observationCount;
    map<int, int> featureCounts;

public:
    TreeNode(int id, int obsCount, const map<int, int>& counts);
    virtual ~TreeNode() = default;

    int getId() const;
    int getObservationCount() const;
    virtual bool isLeaf() const = 0;
    virtual json toJSON() const = 0;
    const map<int, int>& getFeatureCounts() const;
};

class LeafNode : public TreeNode {
private:
    // No additional members needed for LeafNode
    // All data is inherited from TreeNode
public:
    LeafNode(int id, int obsCount, const map<int, int>& counts);
    bool isLeaf() const override;
    json toJSON() const override;
};

class SplitNode : public TreeNode {
private:
    int columnID;
    int leftCount;
    int rightCount;
    set<int> leftSubset;
    set<int> rightSubset;
    unique_ptr<TreeNode> left;
    unique_ptr<TreeNode> right;
    

public:
    SplitNode(int id, int obsCount, int columnID, int leftCount, int rightCount,
              const set<int>& left, const set<int>& right, const map<int, int>& featureCounts);

    bool isLeaf() const override;
    json toJSON() const override;

    void setLeft(unique_ptr<TreeNode> l);
    void setRight(unique_ptr<TreeNode> r);

    TreeNode* getLeft() const;
    TreeNode* getRight() const;

    bool hasLeft() const;
    bool hasRight() const;

    int getColumnID() const;
    int getLeftCount() const;
    int getRightCount() const;
    
    const set<int>& getLeftSubset() const;
    const set<int>& getRightSubset() const;

    void setRightSubset(const std::set<int>& subset);

};

class Tree {
private:
    unique_ptr<TreeNode> root;
    vector<SplitNode*> stack;
    int nextNodeId;
    

    int generateId();

public:
    Tree();
    TreeNode* getRoot() const;

    void writeTreeBinary(std::ostream& out) const;
    explicit Tree(std::istream& in);
    
    void createSplit(const SplitInfo& split);
    void createLeaf(const LeafInfo& leaf);
    int getCurrentNodeId() const;
    int getCurrentColumnId() const;
    bool currentNodeHasLeft() const;
    bool currentNodeHasRight() const;

    bool isComplete();
    void writeTreeJSON(ostream& out, int targetColumn) const;

    void writeTreeDOT(std::ostream& out) const;
    set<int> getCurrentSubset() const;

};

#endif // TREE_H