// src/Threading/ThreadConfig.h
#pragma once
#include <atomic>
#include <algorithm>

inline std::atomic<int>& __GlobalThreadPoolSize() {
    static std::atomic<int> v{20}; // previous default
    return v;
}
inline void SetGlobalThreadPoolSize(int n) {
    __GlobalThreadPoolSize().store(std::max(1, n), std::memory_order_relaxed);
}
inline int GetGlobalThreadPoolSize() {
    return __GlobalThreadPoolSize().load(std::memory_order_relaxed);
}
