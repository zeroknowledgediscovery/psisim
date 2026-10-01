#ifndef SUBSET_SEARCH_CONFIG_H
#define SUBSET_SEARCH_CONFIG_H

#include <stdexcept>
#include <string>

enum class SubsetSearchMode {
    Exact,
    Fast,
    Auto
};

inline SubsetSearchMode& GlobalSubsetSearchModeStorage() {
    static SubsetSearchMode mode = SubsetSearchMode::Exact;
    return mode;
}

inline int& GlobalMaxExactLevelsStorage() {
    static int value = 20;
    return value;
}

inline int& GlobalFastLevelsStorage() {
    static int value = 16;
    return value;
}

inline void SetGlobalSubsetSearchMode(const std::string& mode) {
    if (mode == "exact") {
        GlobalSubsetSearchModeStorage() = SubsetSearchMode::Exact;
    } else if (mode == "fast") {
        GlobalSubsetSearchModeStorage() = SubsetSearchMode::Fast;
    } else if (mode == "auto") {
        GlobalSubsetSearchModeStorage() = SubsetSearchMode::Auto;
    } else {
        throw std::invalid_argument("subset mode must be one of: exact, fast, auto");
    }
}

inline SubsetSearchMode GetGlobalSubsetSearchMode() {
    return GlobalSubsetSearchModeStorage();
}

inline void SetGlobalMaxExactLevels(int value) {
    if (value < 2 || value > 63) {
        throw std::invalid_argument("max exact levels must be between 2 and 63");
    }
    GlobalMaxExactLevelsStorage() = value;
}

inline int GetGlobalMaxExactLevels() {
    return GlobalMaxExactLevelsStorage();
}

inline void SetGlobalFastLevels(int value) {
    if (value < 2 || value > 20) {
        throw std::invalid_argument("fast levels must be between 2 and 20");
    }
    GlobalFastLevelsStorage() = value;
}

inline int GetGlobalFastLevels() {
    return GlobalFastLevelsStorage();
}

#endif
