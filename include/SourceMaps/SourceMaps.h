#ifndef SOURCE_MAPS_H
#define SOURCE_MAPS_H

#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace sourcemaps {

struct ColumnMaps {
    std::vector<std::string> id2str;             
    std::unordered_map<std::string, int> str2id;  
    std::vector<int> t2s;                         
    std::vector<int> s2t;                         
};

class SourceMapStore {
public:
    explicit SourceMapStore(std::filesystem::path runDir, int colsPerShard = 50000);

    static std::filesystem::path inferRunDir(const std::filesystem::path& treesDir);

    const std::filesystem::path& runDir() const { return runDir_; }
    int colsPerShard() const { return colsPerShard_; }

    const ColumnMaps* tryGet(int colId);   
    const ColumnMaps& get(int colId);      

    int encodeToken(int colId, const std::string& token);
    std::vector<int> encodeRow(const std::vector<std::string>& tokens);
    std::string decodeLabel(int colId, int translatedCode);

private:
    bool loadColumn(int colId, ColumnMaps& out);

    std::filesystem::path runDir_;
    int colsPerShard_;
    mutable std::mutex mtx_;
    std::unordered_map<int, ColumnMaps> columnMaps_;
};

}  

#endif  