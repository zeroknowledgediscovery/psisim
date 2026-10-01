// FeatureSubsetSelector.h
#ifndef FEATURE_SUBSET_SELECTOR_H
#define FEATURE_SUBSET_SELECTOR_H

#ifdef USE_WEBSOCKET_LOGGING
#include "WebSocketLogger/WebSocketLogger.h"
#endif

#include <vector>
#include <map>
#include <set>
#include <cmath>

using namespace std;

class FeatureSubsetSelector {
public:
    FeatureSubsetSelector();
    ~FeatureSubsetSelector();

    void setAlpha(double alpha);
    void setTargetColumn(int columnNum);
    void setFeatureColumn(int columnNum);
    void addTargetData(int data);
    void addFeatureData(int data);
    void calculateBestSubset();

    bool isSignificantSubsetFound() const;
    vector<int> getBestSubset() const;
    vector<int> getBestSubsetComplement() const;
    #ifdef USE_WEBSOCKET_LOGGING
    void setWebSocketLogger(WebSocketLogger* logger);
    #endif

private:
    struct Pair { int feature; int target; };
    struct SubsetResult {
        set<int> subset;
        set<int> complement;
        double pValue = 1.0;
        bool isSignificant = false;
        double threshold = 1.0;
        int subsetCount = 0;
        int complementCount = 0;
    };

    int targetColumnNum = -1;
    int featureColumnNum = -1;
    int totalObservations = 0;
    int rowSize = 0;
    int columnSize = 0;
    double alpha;

    vector<int> rawTargetData;
    vector<int> rawFeatureData;
    vector<Pair> filteredPairs;

    map<int, int> targetIndexMap;
    map<int, int> featureIndexMap;

    SubsetResult bestResult;

    void reset();
    void filterValidPairs();
    void calculateSubsetPValue(const set<int>& subset, SubsetResult& result);
    void applyHolmBonferroni(vector<SubsetResult>& allResults);

    #ifdef USE_WEBSOCKET_LOGGING
    WebSocketLogger* webSocketLogger = nullptr;
    int logKey;
    int progressBarId;
    #endif
};

#endif