/*
Instead of duplicating the source map code logic to bindings files, I built a shared one which helps 
- Easier to fix bugs in one place
- More consistent across tools
- Slightly faster because no need to copy ColumnMaps, column caches live in the store
*/
#include "SourceMaps/SourceMaps.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <memory>
#include <stdexcept>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace sourcemaps {
namespace {

json parseJsonFile(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open file: " + path.string());

    const auto size = fs::file_size(path);
    std::string buffer;
    buffer.resize(size ? static_cast<size_t>(size) : 0);
    if (size) file.read(buffer.data(), static_cast<std::streamsize>(size));
    return json::parse(buffer, nullptr, true, true);
}

fs::path shardPathFor(const fs::path& runDir, int colId, int colsPerShard) {
    fs::path per = runDir / "source_maps" / "json_shards" / (std::to_string(colId) + ".json");
    if (fs::exists(per)) return per;
    const int bucket = colId / colsPerShard;
    return runDir / "source_maps" / "json_shards" / (std::to_string(bucket) + ".json");
}

void applyT2S(ColumnMaps& colMaps, std::vector<int> t2s) {
    if (t2s.empty()) return;
    colMaps.t2s = std::move(t2s);

    int maxSid = 0;
    for (int sid : colMaps.t2s) maxSid = std::max(maxSid, sid);

    colMaps.s2t.assign(static_cast<size_t>(maxSid) + 1, 0);
    for (int i = 0; i < static_cast<int>(colMaps.t2s.size()); ++i) {
        const int sid = colMaps.t2s[static_cast<size_t>(i)];
        if (sid >= 0 && sid < static_cast<int>(colMaps.s2t.size())) {
            colMaps.s2t[static_cast<size_t>(sid)] = i;
        }
    }
}

void fillT2SFromJson(ColumnMaps& colMaps, const json& j) {
    std::vector<int> t2s;
    if (j.is_array()) {
        t2s.resize(j.size(), 0);
        for (size_t i = 0; i < j.size(); ++i) t2s[i] = j[i].get<int>();
    } else if (j.is_object()) {
        int maxTid = 0;
        for (auto& [k, _] : j.items()) maxTid = std::max(maxTid, std::stoi(k));
        t2s.assign(static_cast<size_t>(maxTid) + 1, 0);
        for (auto& [k, v] : j.items()) t2s[static_cast<size_t>(std::stoi(k))] = v.get<int>();
    }
    applyT2S(colMaps, std::move(t2s));
}

int tokenToTranslated(const ColumnMaps& colMaps, const std::string& token) {
    const auto it = colMaps.str2id.find(token);
    const int sid = (it == colMaps.str2id.end()) ? 0 : it->second;

    if (!colMaps.s2t.empty()) {
        if (sid >= 0 && sid < static_cast<int>(colMaps.s2t.size())) {
            return colMaps.s2t[static_cast<size_t>(sid)];
        }
        return 0;
    }
    return sid;
}

std::string codeToLabel(const ColumnMaps& cm, int code) {
    const int sid = (!cm.t2s.empty() && code >= 0 && code < static_cast<int>(cm.t2s.size()))
                        ? cm.t2s[static_cast<size_t>(code)]
                        : code;
    if (sid >= 0 && sid < static_cast<int>(cm.id2str.size())) {
        return cm.id2str[static_cast<size_t>(sid)];
    }
    return std::string("<UNK:") + std::to_string(code) + ">";
}

} 

SourceMapStore::SourceMapStore(fs::path runDir, int colsPerShard)
    : runDir_(std::move(runDir)), colsPerShard_(colsPerShard) {}

fs::path SourceMapStore::inferRunDir(const fs::path& treesDir) {
    fs::path p = fs::absolute(treesDir);
    for (int i = 0; i < 4; ++i) {
        if (fs::exists(p / "source_maps")) return p;
        p = p.parent_path();
        if (p.empty()) break;
    }
    return {};
}

bool SourceMapStore::loadColumn(int colId, ColumnMaps& out) {
    const fs::path p = shardPathFor(runDir_, colId, colsPerShard_);
    if (!fs::exists(p)) return false;

    // Do not cache shard DOMs process-wide by pathname. Null calibration
    // deliberately reuses rep_N directories across --overwrite runs, so a
    // pathname-only cache can silently return a previous model's source map.
    // SourceMapStore already caches decoded columns per instance.
    const json J = parseJsonFile(p);
    json node;
    if (J.is_object() && J.contains(std::to_string(colId)))
        node = J[std::to_string(colId)];
    else
        node = J;

    ColumnMaps colMaps;
    if (node.contains("column_strings_map") && node["column_strings_map"].is_array()) {
        const auto& arr = node["column_strings_map"];
        colMaps.id2str.resize(arr.size(), "");
        for (size_t i = 0; i < arr.size(); ++i) {
            colMaps.id2str[i] = arr[i].is_string() ? arr[i].get<std::string>() : "";
        }
    } else if (node.contains("to_str") && node["to_str"].is_object()) {
        int maxId = 0;
        for (auto& [k, _] : node["to_str"].items()) maxId = std::max(maxId, std::stoi(k));
        colMaps.id2str.assign(static_cast<size_t>(maxId) + 1, "");
        for (auto& [k, v] : node["to_str"].items()) {
            const int id = std::stoi(k);
            colMaps.id2str[static_cast<size_t>(id)] = v.is_string() ? v.get<std::string>() : "";
        }
    } else {
        return false;
    }

    colMaps.str2id.clear();
    colMaps.str2id.reserve(colMaps.id2str.size());
    for (int i = 0; i < static_cast<int>(colMaps.id2str.size()); ++i) {
        colMaps.str2id[colMaps.id2str[static_cast<size_t>(i)]] = i;
    }

    if (node.contains("translate") && node["translate"].is_object()) {
        fillT2SFromJson(colMaps, node["translate"]);
    }

    const fs::path t2sPath =
        runDir_ / "source_maps" / "translated_to_source" / (std::to_string(colId) + ".json");
    if (fs::exists(t2sPath)) {
        fillT2SFromJson(colMaps, parseJsonFile(t2sPath));
    }

    out = std::move(colMaps);
    return true;
}

const ColumnMaps* SourceMapStore::tryGet(int colId) {
    {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = columnMaps_.find(colId);
        if (it != columnMaps_.end()) return &it->second;
    }

    ColumnMaps colMaps;
    if (!loadColumn(colId, colMaps)) return nullptr;

    std::lock_guard<std::mutex> lock(mtx_);
    auto [it, inserted] = columnMaps_.emplace(colId, std::move(colMaps));
    (void)inserted;
    return &it->second;
}

const ColumnMaps& SourceMapStore::get(int colId) {
    const ColumnMaps* colMaps = tryGet(colId);
    if (!colMaps) {
        throw std::runtime_error("source_maps missing for column " + std::to_string(colId) +
                                 " under " + runDir_.string());
    }
    return *colMaps;
}

int SourceMapStore::encodeToken(int colId, const std::string& token) {
    const ColumnMaps* colMaps = tryGet(colId);
    if (!colMaps) return 0;
    return tokenToTranslated(*colMaps, token);
}

std::vector<int> SourceMapStore::encodeRow(const std::vector<std::string>& tokens) {
    std::vector<int> codes(tokens.size(), 0);
    for (size_t col = 0; col < tokens.size(); ++col) {
        codes[col] = encodeToken(static_cast<int>(col), tokens[col]);
    }
    return codes;
}

std::string SourceMapStore::decodeLabel(int colId, int translatedCode) {
    const ColumnMaps* cm = tryGet(colId);
    if (!cm) return std::string("<UNK:") + std::to_string(translatedCode) + ">";
    return codeToLabel(*cm, translatedCode);
}

}  
