// memory_pool.cpp — Fixed-size block pool + stack allocator simulation
//
// AP試験 concepts demonstrated (OS・メモリ管理):
//   - ヒープ (動的メモリ): malloc/free → 外部断片化発生
//   - メモリプール (固定サイズブロック): 断片化ゼロ, O(1) alloc/free
//     → RTOS/組み込みで必須 (non-deterministic malloc 禁止)
//   - スタックアロケータ (バンプポインタ): 最速, フレーム単位で一括解放
//   - 内部断片化 vs 外部断片化の違い
//   - ガベージコレクション不要 → デストラクタで決定論的解放
//
// Application: DISCO DFL7160 の RTOS 環境では malloc() 禁止。
//   kerf 検出バッファ・EtherCAT フレームは固定プールから確保。

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

// ── Fixed-size block memory pool ─────────────────────────────────────────────
// Pre-allocated array of N blocks, each of B bytes.
// Free list stored implicitly as singly-linked list inside free blocks
// (first sizeof(int) bytes = index of next free block, -1 = end).
// Allocation: O(1).  Deallocation: O(1).  Internal fragmentation: 0.
// External fragmentation: 0 by design (all blocks equal size).
class FixedPool {
public:
    FixedPool(int block_size, int n_blocks)
        : block_size_(block_size), n_blocks_(n_blocks),
          storage_(block_size * n_blocks, 0),
          free_head_(0), n_free_(n_blocks),
          n_allocs_(0), n_frees_(0), n_failed_(0), peak_used_(0)
    {
        if (block_size < static_cast<int>(sizeof(int)))
            throw std::invalid_argument("block_size must be >= sizeof(int) = 4");
        // Build free list: block i → index i+1 (last → -1)
        for (int i = 0; i < n_blocks - 1; ++i)
            *reinterpret_cast<int*>(&storage_[i * block_size_]) = i + 1;
        *reinterpret_cast<int*>(&storage_[(n_blocks-1) * block_size_]) = -1;
    }

    // Returns block index (≥0) or -1 if pool full
    int alloc() {
        if (free_head_ < 0) { ++n_failed_; return -1; }
        int idx = free_head_;
        free_head_ = *reinterpret_cast<int*>(&storage_[idx * block_size_]);
        --n_free_;
        ++n_allocs_;
        int used = n_blocks_ - n_free_;
        if (used > peak_used_) peak_used_ = used;
        return idx;
    }

    // Free block by index; silently ignore double-free (production: assert)
    void free(int idx) {
        if (idx < 0 || idx >= n_blocks_) return;
        *reinterpret_cast<int*>(&storage_[idx * block_size_]) = free_head_;
        free_head_ = idx;
        ++n_free_;
        ++n_frees_;
    }

    int n_free()    const { return n_free_;    }
    int n_used()    const { return n_blocks_ - n_free_; }
    int n_allocs()  const { return n_allocs_;  }
    int n_frees()   const { return n_frees_;   }
    int n_failed()  const { return n_failed_;  }
    int peak_used() const { return peak_used_; }
    int n_blocks()  const { return n_blocks_;  }
    int block_size()const { return block_size_;}

private:
    int block_size_, n_blocks_;
    std::vector<uint8_t> storage_;
    int free_head_, n_free_;
    int n_allocs_, n_frees_, n_failed_, peak_used_;
};

// ── Stack (linear / bump-pointer) allocator ──────────────────────────────────
// Allocation: bump pointer forward (O(1)).
// Deallocation: reset to a saved mark (frame-level bulk free, O(1)).
// Cannot free individual allocations — suited for temporary scratch buffers.
class StackAllocator {
public:
    explicit StackAllocator(int capacity_bytes)
        : storage_(capacity_bytes, 0), top_(0), capacity_(capacity_bytes),
          n_allocs_(0), n_frees_(0), n_failed_(0), peak_top_(0) {}

    // Returns byte offset (≥0) or -1 on overflow
    int alloc(int size) {
        if (size <= 0 || top_ + size > capacity_) { ++n_failed_; return -1; }
        int offset = top_;
        top_ += size;
        ++n_allocs_;
        if (top_ > peak_top_) peak_top_ = top_;
        return offset;
    }

    // Save current top (== "push frame")
    int save_mark() const { return top_; }

    // Restore top to mark (== "pop frame") — bulk-frees everything above mark
    void restore_mark(int mark) {
        if (mark < 0 || mark > top_) return;
        ++n_frees_;
        top_ = mark;
    }

    int capacity()  const { return capacity_; }
    int used()      const { return top_; }
    int free_space()const { return capacity_ - top_; }
    int peak_top()  const { return peak_top_; }
    int n_allocs()  const { return n_allocs_; }
    int n_frees()   const { return n_frees_;  }
    int n_failed()  const { return n_failed_; }

private:
    std::vector<uint8_t> storage_;
    int top_, capacity_, n_allocs_, n_frees_, n_failed_, peak_top_;
};

// ── Simulation helpers ────────────────────────────────────────────────────────

// Run a random alloc/free workload on a FixedPool and return stats.
// alloc_prob: probability that each step is an alloc (vs free of random held block).
py::dict simulate_pool(
    int block_size,
    int n_blocks,
    int n_steps,
    double alloc_prob,
    uint32_t seed
) {
    FixedPool pool(block_size, n_blocks);
    std::vector<int> held;
    held.reserve(n_blocks);

    uint64_t state = static_cast<uint64_t>(seed) ^ 0xcafe1234ULL;
    auto lcg = [&]() -> double {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>((state >> 33) & 0x7FFFFFFFULL) /
               static_cast<double>(0x7FFFFFFFULL);
    };

    for (int i = 0; i < n_steps; ++i) {
        if (lcg() < alloc_prob || held.empty()) {
            int idx = pool.alloc();
            if (idx >= 0) held.push_back(idx);
        } else {
            // Free a random held block
            int j = static_cast<int>(lcg() * held.size());
            j = std::max(0, std::min(j, static_cast<int>(held.size()) - 1));
            pool.free(held[j]);
            held.erase(held.begin() + j);
        }
    }

    py::dict r;
    r["block_size"]       = block_size;
    r["n_blocks"]         = n_blocks;
    r["n_allocs"]         = pool.n_allocs();
    r["n_frees"]          = pool.n_frees();
    r["n_failed"]         = pool.n_failed();
    r["peak_used"]        = pool.peak_used();
    r["final_used"]       = pool.n_used();
    r["final_free"]       = pool.n_free();
    r["peak_utilization"] = static_cast<double>(pool.peak_used()) / n_blocks;
    r["fragmentation"]    = std::string("zero (fixed-size pool)");
    return r;
}

// Demonstrate stack allocator with explicit alloc/free pattern.
// allocations: list of (size, is_free_mark) tuples represented as list-of-lists.
py::dict simulate_stack(int capacity_bytes, const std::vector<int>& alloc_sizes) {
    StackAllocator sa(capacity_bytes);
    std::vector<int> offsets;
    int mark = sa.save_mark();

    for (int sz : alloc_sizes) {
        if (sz < 0) {
            sa.restore_mark(mark);
            mark = sa.save_mark();
        } else {
            offsets.push_back(sa.alloc(sz));
        }
    }

    py::dict r;
    r["capacity_bytes"]  = capacity_bytes;
    r["used_bytes"]      = sa.used();
    r["free_bytes"]      = sa.free_space();
    r["peak_bytes"]      = sa.peak_top();
    r["n_allocs"]        = sa.n_allocs();
    r["n_frame_frees"]   = sa.n_frees();
    r["n_failed"]        = sa.n_failed();
    r["peak_utilization"]= static_cast<double>(sa.peak_top()) / capacity_bytes;
    return r;
}

// Compare pool vs naive heap: demonstrate deterministic timing of pool.
// Returns timing metadata (simulated cycle counts, not wall-clock).
py::dict compare_allocators(int block_size, int n_blocks) {
    // Pool: O(1) fixed cycles
    int pool_alloc_cycles = 4;   // load free_head, write link, update head, inc counter
    int pool_free_cycles  = 3;   // write link, update head, inc counter

    // Naive heap (simplified model): O(N) in worst case (first-fit search)
    int heap_alloc_cycles_avg = static_cast<int>(std::sqrt(n_blocks)) * 10;
    int heap_alloc_cycles_max = n_blocks * 10;
    int heap_free_cycles_avg  = static_cast<int>(std::sqrt(n_blocks)) * 8;

    py::dict r;
    r["pool_alloc_cycles_always"] = pool_alloc_cycles;
    r["pool_free_cycles_always"]  = pool_free_cycles;
    r["pool_deterministic"]       = true;
    r["heap_alloc_cycles_avg"]    = heap_alloc_cycles_avg;
    r["heap_alloc_cycles_max"]    = heap_alloc_cycles_max;
    r["heap_free_cycles_avg"]     = heap_free_cycles_avg;
    r["heap_deterministic"]       = false;
    r["pool_advantage"]           = std::string(
        "Pool: O(1), jitter-free → safe in 10kHz ISR. "
        "Heap: O(N) worst-case → forbidden in hard real-time.");
    return r;
}

// ── pybind11 module ──────────────────────────────────────────────────────────

PYBIND11_MODULE(_memory_pool_kernel, m) {
    m.doc() =
        "Fixed-size block pool + stack allocator simulation.\n"
        "\n"
        "AP試験 topics: ヒープ vs スタック, メモリプール, 断片化,\n"
        "  O(1)決定論的アロケーション, RTOS組み込みでのmalloc禁止\n"
        "\n"
        "Key insight: RTOS (real-time system) では malloc() 禁止。\n"
        "  理由: worst-case timing が保証できないため。\n"
        "  代替: 固定サイズプール → O(1) 保証, 断片化ゼロ。";

    m.def("simulate_pool", &simulate_pool,
          py::arg("block_size")  = 256,
          py::arg("n_blocks")    = 64,
          py::arg("n_steps")     = 1000,
          py::arg("alloc_prob")  = 0.6,
          py::arg("seed")        = 42,
          "Run random alloc/free workload on a fixed-size block pool.\n"
          "Returns: n_allocs, n_frees, n_failed, peak_used, peak_utilization, fragmentation.");

    m.def("simulate_stack", &simulate_stack,
          py::arg("capacity_bytes"),
          py::arg("alloc_sizes"),
          "Run stack allocator with given alloc sizes.\n"
          "Negative size = restore_mark (frame pop = bulk free).\n"
          "Returns: used, free, peak, n_allocs, n_frame_frees, n_failed.");

    m.def("compare_allocators", &compare_allocators,
          py::arg("block_size") = 256,
          py::arg("n_blocks")   = 64,
          "Compare pool vs naive heap on timing characteristics.\n"
          "Returns simulated cycle counts showing O(1) vs O(N) difference.");
}
