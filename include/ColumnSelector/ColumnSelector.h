// ColumnSelector.h
#ifndef COLUMN_SELECTOR_H
#define COLUMN_SELECTOR_H

#include <vector>
#include <map>
#include <set>
#include <limits>
#include <stdexcept>
#include <boost/math/distributions/chi_squared.hpp>

using namespace std;
using namespace boost::math;

struct ChiSquareData {
    int featureColumn;
    double chiSquare;
    double pValue;
    int degreeOfFreedom;
    int columnCount;
    int rowCount;
    int totalObservations;
};

class ColumnSelector {
private:
    int targetColumnNum = -1;
    int featureColumnNum = -1;
    int totalObservations = 0;

    double alpha;

    vector<int> rawTargetData;
    vector<int> rawFeatureData;

    int** contingencyTable = nullptr;
    int* rowTotals = nullptr;
    int* columnTotals = nullptr;
    int rowSize = 0;
    int columnSize = 0;

    vector<ChiSquareData> chiSquareDataList;
    vector<int> significantColumns;

    void allocateTable(int rowCount, int colCount);
    void clearTable();
    ChiSquareData computeChiSquare();
    void applyHolmBonferroni();

public:
    ColumnSelector();
    ~ColumnSelector();

    void setAlpha(double alpha);
    void setTargetColumn(int columnNum);
    void setFeatureColumn(int columnNum);
    void addTargetData(int data);
    void addFeatureData(int data);
    void processColumn();
    vector<int> getSignificantColumns();
};

#endif