#include "ShardMap/ShardMap.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include <vector>
#include <cctype>
#include <mutex>
#include <thread>

namespace shardio {

// Parser workers and concurrent null fits can update shard files at the same
// time. Protect the complete read-modify-write transaction; otherwise another
// reader can observe the truncate window of std::ofstream and parse an empty
// JSON document.
static std::mutex shard_write_mutex;

static inline std::filesystem::path shard_path(const std::filesystem::path& root, std::uint64_t column_id) {
    const auto shard_id = column_id / COLUMNS_PER_SHARD;
    return root / (std::to_string(shard_id) + ".json"); 
}

ShardMap::ShardMap(const std::filesystem::path& rootDir) : rootDir_(rootDir) {}
void ShardMap::setRootDirectory(const std::filesystem::path& newRootDir) { rootDir_ = newRootDir; }

//normalization + hash (platform-independent)
std::string ShardMap::normalizeHeader(const std::string& s) {
    std::string out; out.reserve(s.size());
    // ASCII lowercase + trim (simpple case-insensitive normalization for headers)
    size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e-1]))) --e;
    out.resize(e - b);
    for (size_t i = 0; i < out.size(); ++i) {
        unsigned char ch = static_cast<unsigned char>(s[b + i]);
        out[i] = static_cast<char>(std::tolower(ch));
    }
    return out;
}

std::uint64_t ShardMap::fnv1a64(const std::string& s) {
    const std::uint64_t FNV_OFFSET = 1469598103934665603ull;
    const std::uint64_t FNV_PRIME  = 1099511628211ull;
    std::uint64_t h = FNV_OFFSET;
    for (unsigned char c : s) {
        h ^= c;
        h *= FNV_PRIME;
    }
    return h;
}

void ShardMap::appendHeaderIndexRecord(const std::string& header,
                                       const std::string& norm,
                                       std::uint64_t column_id)
{
    std::filesystem::path idxDir = rootDir_ / "header_index";
    std::error_code ec;
    std::filesystem::create_directories(idxDir, ec);

    std::uint64_t h = fnv1a64(norm);
    std::uint64_t bucket = h & (HEADER_INDEX_BUCKETS - 1); // power-of-two mask
    auto bucketFile = idxDir / (std::to_string(bucket) + ".jsonl");

    nlohmann::json rec;
    rec["header"] = header;  
    rec["norm"]   = norm;     // normalized (for case-insensitive lookup)
    rec["col"]    = column_id;

    std::ofstream out(bucketFile, std::ios::binary | std::ios::app);
    out << rec.dump() << '\n';
}



// --- header-aware append: writes shard + header index line ---
void ShardMap::appendWithHeader(std::uint64_t column_id,
                                const std::unordered_map<std::string,int>& str_to_int,
                                const std::string& column_header) {
    // Build the data-code -> string array.  The parser has already excluded
    // the CSV header row from str_to_int, so a data symbol whose text happens
    // to equal column_header is still a legitimate category and must remain.
    size_t maxc = 0;
    for (const auto& kv : str_to_int) if (kv.second >= 0) maxc = std::max(maxc, static_cast<size_t>(kv.second));

    std::vector<nlohmann::json> arr(maxc + 1, nullptr);
    if (!arr.empty()) arr[0] = ""; // 0 is reserved for empty string

    for (const auto& [s, id] : str_to_int) {
        if (id < 0) continue;
        if (static_cast<size_t>(id) < arr.size()) arr[static_cast<size_t>(id)] = s;
    }

    nlohmann::json labels = std::move(arr);

    // Upsert into shard.  This must be one synchronized transaction:
    // multiple parser workers may target columns in the same JSON shard.
    std::error_code ec; std::filesystem::create_directories(rootDir_, ec);
    auto shardFile = shard_path(rootDir_, column_id);

    {
        std::lock_guard<std::mutex> lock(shard_write_mutex);
        nlohmann::json shardJson;
        if (std::filesystem::exists(shardFile)) {
            std::ifstream in(shardFile, std::ios::binary);
            if (in.peek() != std::ifstream::traits_type::eof()) in >> shardJson;
        }

        nlohmann::json colObj;
        colObj["column_header"]      = column_header;
        colObj["column_strings_map"] = labels;
        shardJson[std::to_string(column_id)] = std::move(colObj);

        // Write a complete temporary document, flush/close it, then publish by
        // rename. Readers therefore never see a zero-length/partial shard.
        auto tmpFile = shardFile;
        tmpFile += ".tmp." + std::to_string(
            std::hash<std::thread::id>{}(std::this_thread::get_id()));
        {
            std::ofstream out(tmpFile, std::ios::binary | std::ios::trunc);
            if (!out) throw std::runtime_error("cannot write shard: " + tmpFile.string());
            out << shardJson.dump();
            out.flush();
            if (!out) throw std::runtime_error("failed writing shard: " + tmpFile.string());
        }
        std::filesystem::rename(tmpFile, shardFile, ec);
        if (ec) {
            std::filesystem::remove(tmpFile);
            throw std::runtime_error("cannot publish shard: " + ec.message());
        }

        // Keep the header index append under the same lock as well.
        const std::string norm = normalizeHeader(column_header);
        appendHeaderIndexRecord(column_header, norm, column_id);
    }
}

} // namespace shardio
