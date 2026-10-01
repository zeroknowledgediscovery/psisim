#ifndef SHARD_MAP_HPP
#define SHARD_MAP_HPP

// ShardMap.hpp
#include <string>
#include <unordered_map>
#include <vector>
#include <filesystem>

#ifndef COLUMNS_PER_SHARD
#define COLUMNS_PER_SHARD 50000
#endif

#ifndef SHARD_SUBDIR
#define SHARD_SUBDIR "source_maps/json_shards"
#endif

#ifndef HEADER_INDEX_BUCKETS
// power of two for fast masking 
#define HEADER_INDEX_BUCKETS 4096
#endif

namespace shardio {

class ShardMap {
public:
    explicit ShardMap(const std::filesystem::path& rootDir);
    void setRootDirectory(const std::filesystem::path& newRootDir);

    // Your existing APIs
    void append(std::uint64_t column_id,
                const std::unordered_map<std::string,int>& str_to_int);

    // Writes: { "column_header": <header>, "column_strings_map": [...] }
    void appendWithHeader(std::uint64_t column_id,
                          const std::unordered_map<std::string,int>& str_to_int,
                          const std::string& column_header);

private:
    std::filesystem::path rootDir_;

    static std::string normalizeHeader(const std::string& s);
    static std::uint64_t fnv1a64(const std::string& s);
    void appendHeaderIndexRecord(const std::string& header,
                                 const std::string& norm,
                                 std::uint64_t column_id);
};

} // namespace shardio



#endif // SHARD_MAP_HPP