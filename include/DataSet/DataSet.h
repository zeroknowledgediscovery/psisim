#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <set>
#include <map>
#include <string>
#include <cstdint>

using namespace std;

typedef struct ColumnGroupData {
  int numColumns;
  int bitWidth;
  int startColumn;
  int endColumn;
} ColumnGroupData;


typedef struct FileData {
  int numColumnGroups;
  ColumnGroupData *columnGroups;
} FileData;

typedef struct ColumnAddress {
  int fileIndex;
  int columnGroupIndex;
  int columnIndex;
  int bitWidth;
  uint64_t byteOffset;
  int bitOffset;
} ColumnAddress;

class DataSet {
private:
  struct RoutePlan {
    int featureColumn;
    set<int> leftSubset;
    set<int> rightSubset;
    bool rightSubsetKnown = false;
    double piLeft = 0.5;
  };

  int numFiles;
  int totalRows;
  int totalColumns;
  int* enabledRows;
  bool targetMaskInitialized;
  int targetColumnId;
  string dataDirectory;
  FileData* fileData;

  vector<vector<int>> columnCache;
  vector<unsigned char> columnCacheReady;

  // Keyed by the existing DFS row-routing token (split node ID + 1).
  // The same token is used for the left and right visits of one split, while
  // every split has a distinct node ID. This makes sibling routing reuse exact
  // and independent of the current active-row hash.
  map<int, RoutePlan> routePlans;

  const vector<int>& getColumnDataCached(int columnIndex);
  static uint64_t splitmix64(uint64_t x);
  bool missingGoesLeft(int routeToken, const RoutePlan& plan, int rowIndex) const;
  void routeRows(int routeToken, int featureColumn, const std::set<int>& featureValues);
  void routeRows(int routeToken, int featureColumn, int featureValue);
    
public:
  DataSet(const std::string& dataDirectory, int totalRows, int totalColumns, int numFiles);
  ~DataSet();

  // Missing target is absence of an outcome for this target tree and is
  // permanently excluded from that tree's construction.
  void setTargetColumn(int targetColumn);

  void enableRows(int nodeID);
  void disableRows(int nodeID, int featureColumn, int featureValue);
  void disableRows(int nodeID, int featureColumn, const std::set<int>& featureValues);

  bool isRowEnabled(int rowIndex) const;
  int getTotalEnabledRows() const;

  void loadColumnData(int columnIndex, int* data) ;
  ColumnAddress findColumnAddress(int columnIndex);
  vector<int> buildColumnData(int columnIndex) ;

  void buildFileData();
  void getColumnDist(int columnIndex, map<int,int>& columnDist, bool skipMissing = true);
};

#endif // DATASET_H
