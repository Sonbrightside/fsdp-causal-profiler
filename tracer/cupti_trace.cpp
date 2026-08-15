#include <cupti.h>
#include <nvtx3/nvToolsExt.h>
#include <generated_nvtx_meta.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
//#include <generated_nvtx_meta.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <sys/syscall.h>
#include <unistd.h>
#include <thread>
#include <condition_variable>
#include <chrono>
#include <algorithm>   // std::find
#include <sqlite3.h>
#include <cuda.h>
#include <iterator>   // std::make_move_iterator
#include <utility>    // std::move
#include "clock_anchor.h"



static std::string dbPathForRank() {
    // 1) Determine rank: prefer RANK set by torchrun; fall back to SLURM_PROCID for direct srun launches
    const char* r = getenv("RANK");
    if (r == nullptr || r[0] == '\0') {
        r = getenv("SLURM_PROCID");
    }
    std::string rank = (r != nullptr && r[0] != '\0') ? r : "0";

    // 2) Output directory: CUPTI_TRACE_DIR if set (node-local /tmp recommended), otherwise cwd
    const char* dir = getenv("CUPTI_TRACE_DIR");
    std::string base = (dir != nullptr && dir[0] != '\0') ? dir : ".";

    return base + "/cupti_trace_rank" + rank + ".sqlite";
}

static const bool g_verbose = (getenv("CUPTI_TRACE_VERBOSE") != nullptr);

#define TRACE_LOG(...)                                  \
    do {                                                \
        if (g_verbose) {                                \
            fprintf(stderr, __VA_ARGS__);               \
            fflush(stderr);                             \
        }                                               \
    } while (0)

// API callback local Buffer size 
static constexpr size_t API_LOCAL_FLUSH_N  = 512;
static constexpr size_t NVTX_LOCAL_FLUSH_N = 128;

// API callback global buffer size
static constexpr size_t API_GLOBAL_MAX  = 1 << 20;  // 1,048,576 records
static constexpr size_t NVTX_GLOBAL_MAX = 1 << 17;  // 131,072 records


// CUPTI Activity buffer
static constexpr size_t ACTIVITY_BUFFER_SIZE = 8 * 1024 * 1024;  // 8 MB
static constexpr size_t ACTIVITY_BUFFER_ALIGNMENT = 8;


//CUPTI Activity global queue maximum record counts
static constexpr size_t KERNEL_ACTIVITY_GLOBAL_MAX = 1 << 18; // 262,144 records
static constexpr size_t MEMCPY_ACTIVITY_GLOBAL_MAX = 1 << 18;


static std::atomic<uint64_t> g_activity_buffers_requested{0};
static std::atomic<uint64_t> g_activity_buffers_completed{0};

struct KernelActivityRecord {
    std::string name;

    uint32_t correlation_id;
    uint32_t stream_id;
    uint32_t device_id;
    uint32_t context_id;

    uint64_t start_ns;
    uint64_t end_ns;
    uint64_t duration_ns;
    uint64_t completed_ns;

    int32_t grid_x;
    int32_t grid_y;
    int32_t grid_z;

    int32_t block_x;
    int32_t block_y;
    int32_t block_z;
};


struct MemcpyActivityRecord {
    uint8_t copy_kind;
    uint64_t bytes;

    uint32_t correlation_id;
    uint32_t runtime_correlation_id;
    uint32_t stream_id;
    uint32_t device_id;
    uint32_t context_id;

    uint64_t start_ns;
    uint64_t end_ns;
    uint64_t duration_ns;
};



static std::mutex g_activity_mutex;

static std::vector<KernelActivityRecord> g_kernel_activity_queue;
static std::vector<MemcpyActivityRecord> g_memcpy_activity_queue;

static std::atomic<uint64_t> g_dropped_kernel_activity{0};
static std::atomic<uint64_t> g_dropped_memcpy_activity{0};


static std::atomic<uint64_t> g_writer_kernel_total{0};
static std::atomic<uint64_t> g_writer_memcpy_total{0};


static void flushActivityBatchesToGlobal(
    std::vector<KernelActivityRecord>& kernel_batch,
    std::vector<MemcpyActivityRecord>& memcpy_batch
) {
    if (kernel_batch.empty() && memcpy_batch.empty()) {
        return;
    }

    size_t dropped_kernel_now = 0;
    size_t dropped_memcpy_now = 0;

    {
        std::lock_guard<std::mutex> lock(g_activity_mutex);

        // ==========================================================
        // Kernel activity queue
        // ==========================================================

        size_t available_kernel_slots = 0;

        if (g_kernel_activity_queue.size() < KERNEL_ACTIVITY_GLOBAL_MAX) {

            available_kernel_slots = KERNEL_ACTIVITY_GLOBAL_MAX - g_kernel_activity_queue.size();
        }

        const size_t accepted_kernel_count = std::min(available_kernel_slots,kernel_batch.size());

        g_kernel_activity_queue.insert(
            g_kernel_activity_queue.end(),

            std::make_move_iterator(
                kernel_batch.begin()
            ),

            std::make_move_iterator(
                kernel_batch.begin() + accepted_kernel_count
            )
        );

        dropped_kernel_now = kernel_batch.size() - accepted_kernel_count;

        // ==========================================================
        // Memcpy activity queue
        // ==========================================================

        size_t available_memcpy_slots = 0;

        if (g_memcpy_activity_queue.size() < MEMCPY_ACTIVITY_GLOBAL_MAX) {

            available_memcpy_slots = MEMCPY_ACTIVITY_GLOBAL_MAX - g_memcpy_activity_queue.size();
        }

        const size_t accepted_memcpy_count = std::min(available_memcpy_slots,memcpy_batch.size());

        g_memcpy_activity_queue.insert(
            g_memcpy_activity_queue.end(),

            std::make_move_iterator(
                memcpy_batch.begin()
            ),

            std::make_move_iterator(
                memcpy_batch.begin() + accepted_memcpy_count
            )
        );

        dropped_memcpy_now = memcpy_batch.size() - accepted_memcpy_count;
    }

    // Local batches are no longer used past this point
    kernel_batch.clear();
    memcpy_batch.clear();

    if (dropped_kernel_now > 0) {
        g_dropped_kernel_activity.fetch_add(
            dropped_kernel_now,
            std::memory_order_relaxed
        );
    }

    if (dropped_memcpy_now > 0) {
        g_dropped_memcpy_activity.fetch_add(
            dropped_memcpy_now,
            std::memory_order_relaxed
        );
    }

    if (dropped_kernel_now > 0 || dropped_memcpy_now > 0) {
        fprintf(stderr,
                "[Activity Queue Overflow] "
                "dropped kernel=%zu memcpy=%zu | "
                "total kernel=%llu memcpy=%llu\n",
                dropped_kernel_now,
                dropped_memcpy_now,
                static_cast<unsigned long long>(
                    g_dropped_kernel_activity.load(
                        std::memory_order_relaxed
                    )
                ),
                static_cast<unsigned long long>(
                    g_dropped_memcpy_activity.load(
                        std::memory_order_relaxed
                    )
                ));

        fflush(stderr);
    }
}

static const char* memcpyKindToString(uint8_t kind) {

    switch (kind) {
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
            return "HtoD";
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
            return "DtoH";
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
            return "HtoA";
        case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
            return "AtoH";
        case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
            return "AtoA";
        case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
            return "AtoD";
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
            return "DtoA";
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
            return "DtoD";
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
            return "HtoH";
        case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP:
            return "PtoP";
        default:
            return "Unknown";
    }
}

static void CUPTIAPI activityBufferRequested(
    uint8_t** buffer,
    size_t* size,
    size_t* maxNumRecords
) {
    if (buffer == nullptr || size == nullptr || maxNumRecords == nullptr) {
        fprintf(stderr, "[CUPTI Activity] bufferRequested got null argument\n");
        fflush(stderr);
        return;
    }

    *size = ACTIVITY_BUFFER_SIZE;
    *maxNumRecords = 0;  // 0 means CUPTI can put as many records as fit

    void* ptr = nullptr;

    int rc = posix_memalign(
        &ptr,
        ACTIVITY_BUFFER_ALIGNMENT,
        ACTIVITY_BUFFER_SIZE
    );

    if (rc != 0 || ptr == nullptr) {
        fprintf(stderr,
                "[CUPTI Activity] failed to allocate activity buffer, rc=%d\n",
                rc);
        fflush(stderr);

        *buffer = nullptr;
        *size = 0;
        return;
    }

    *buffer = static_cast<uint8_t*>(ptr);

    uint64_t count =
        g_activity_buffers_requested.fetch_add(1, std::memory_order_relaxed) + 1;

    TRACE_LOG(
            "[CUPTI Activity] buffer requested: ptr=%p size=%zu count=%llu\n",
            static_cast<void*>(*buffer), *size, (unsigned long long)count);
}


static void CUPTIAPI activityBufferCompleted(
    CUcontext ctx,
    uint32_t streamId,
    uint8_t* buffer,
    size_t size,
    size_t validSize
) {
    uint64_t count =
        g_activity_buffers_completed.fetch_add(1, std::memory_order_relaxed) + 1;

    fprintf(stderr,
            "[CUPTI Activity] buffer completed: ptr=%p size=%zu validSize=%zu "
            "streamId=%u count=%llu\n",
            static_cast<void*>(buffer),
            size,
            validSize,
            streamId,
            (unsigned long long)count);
    fflush(stderr);

    if (buffer == nullptr) {
        return;
    }

    if (validSize == 0) {
        free(buffer);
        return;
    }

        // Callback-local batches
    std::vector<KernelActivityRecord> kernel_batch;
    std::vector<MemcpyActivityRecord> memcpy_batch;


    kernel_batch.reserve(1024);
    memcpy_batch.reserve(256);

    CUpti_Activity* record = nullptr;
    

    while (cuptiActivityGetNextRecord(buffer, validSize, &record) == CUPTI_SUCCESS) {
        
        if (record == nullptr) {
            continue;
        }

        switch (record->kind) {

            case CUPTI_ACTIVITY_KIND_MEMCPY: {
                auto* m = reinterpret_cast<CUpti_ActivityMemcpy*>(record);

               

                MemcpyActivityRecord rec{};

                rec.copy_kind = m->copyKind;
                rec.bytes = m->bytes;

                rec.correlation_id = m->correlationId;
                rec.runtime_correlation_id =
                    m->runtimeCorrelationId;

                rec.stream_id = m->streamId;
                rec.device_id = m->deviceId;
                rec.context_id = m->contextId;

                rec.start_ns = m->start;
                rec.end_ns = m->end;

                rec.duration_ns = m->end >= m->start ? m->end - m->start : 0;

                memcpy_batch.push_back(std::move(rec));

                break;
            }
            case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
                auto* k = reinterpret_cast<CUpti_ActivityKernel9*>(record);

                KernelActivityRecord rec{};

                rec.name = k->name != nullptr ? k->name : "<null>";

                rec.correlation_id = k->correlationId;
                rec.stream_id = k->streamId;
                rec.device_id = k->deviceId;
                rec.context_id = k->contextId;

                rec.start_ns = k->start;
                rec.end_ns = k->end;

                rec.duration_ns = k->end >= k->start ? k->end - k->start : 0;

                rec.completed_ns = k->completed;

                rec.grid_x = k->gridX;
                rec.grid_y = k->gridY;
                rec.grid_z = k->gridZ;

                rec.block_x = k->blockX;
                rec.block_y = k->blockY;
                rec.block_z = k->blockZ;

                kernel_batch.push_back(std::move(rec));

                break;
            }

            default:
                break;
        }
    }

        // Count before flush because flush clears local vectors
    const size_t parsed_kernel_count = kernel_batch.size();

    const size_t parsed_memcpy_count = memcpy_batch.size();
        
                // Local batches → global queues
    flushActivityBatchesToGlobal(kernel_batch, memcpy_batch);

    fprintf(stderr, "[CUPTI Activity] queued kernel=%zu memcpy=%zu\n",
            parsed_kernel_count,
            parsed_memcpy_count);

    fflush(stderr);


    size_t dropped = 0;
    CUptiResult drop_res = cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped);

    if (drop_res == CUPTI_SUCCESS && dropped > 0) {
        fprintf(stderr,
                "[CUPTI Activity] dropped records: %zu\n",
                dropped);
        fflush(stderr);
    }

    free(buffer);
}

static void initActivityTracing() {
    CUptiResult res;

    res = cuptiActivityRegisterCallbacks(
        activityBufferRequested,
        activityBufferCompleted
    );

    if (res != CUPTI_SUCCESS) {
        const char* errstr = nullptr;
        cuptiGetResultString(res, &errstr);

        fprintf(stderr,
                "[CUPTI Activity] cuptiActivityRegisterCallbacks failed: %s\n",
                errstr ? errstr : "unknown");
        fflush(stderr);
        return;
    }

    fprintf(stderr, "[CUPTI Activity] callbacks registered\n");
    fflush(stderr);

    res = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);

        if (res != CUPTI_SUCCESS) {
            const char* errstr = nullptr;
            cuptiGetResultString(res, &errstr);
            fprintf(stderr,
                    "[CUPTI Activity] enable MEMCPY failed: %s\n",
                    errstr ? errstr : "unknown");
            fflush(stderr);
            return;
        }

    fprintf(stderr, "[CUPTI Activity] MEMCPY enabled\n");
    fflush(stderr);

    res = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);

    if (res != CUPTI_SUCCESS) {
        const char* errstr = nullptr;
        cuptiGetResultString(res, &errstr);

        fprintf(stderr,
                "[CUPTI Activity] enable CONCURRENT_KERNEL failed: %s\n",
                errstr ? errstr : "unknown");
        fflush(stderr);
        return;
    }

    fprintf(stderr, "[CUPTI Activity] CONCURRENT_KERNEL enabled\n");
    fflush(stderr);

}




//MACRO for debugging
#define CHECK_CUPTI(call)                                               \
do {                                                                    \
    CUptiResult err = call;                                             \
    if (err != CUPTI_SUCCESS) {                                         \
        const char *errstr = nullptr;                                   \
        cuptiGetResultString(err, &errstr);                             \
        fprintf(stderr, "CUPTI error %s:%d: %s\n",                     \
                __FILE__, __LINE__, errstr ? errstr : "UNKNOWN");       \
        fflush(stderr);                                                 \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
} while (0)



struct ApiEnterInfo {
    uint64_t api_id;
    uint64_t active_nvtx_id;
    uint64_t enter_ns;
};

struct ApiRecord {
    uint64_t api_id;
    uint32_t correlation_id;
    uint32_t domain;
    uint32_t cbid;

    uint64_t active_nvtx_id;
    uint64_t thread_id;

    uint64_t enter_ns;
    uint64_t exit_ns;
    uint64_t duration_ns;

    const char* func_name;
};


struct NvtxFrame {
    uint64_t id;

    uint64_t parent_id;
    uint64_t logical_parent_id;

    uint32_t depth;
    uint64_t thread_id;
    uint64_t push_ns;
    
    std::string name;
};

struct NvtxRangeRecord {
    uint64_t nvtx_id;
    uint64_t parent_id;
    uint32_t depth;
    uint64_t thread_id;

    uint64_t push_ns;
    uint64_t pop_ns;

    std::string name;
};

static void flushApiVectorToGlobal(std::vector<ApiRecord>& local);
static void flushNvtxVectorToGlobal(std::vector<NvtxRangeRecord>& local);



struct ThreadLocalBuffers;



// Global registry for all thread-local buffers
static std::mutex g_tls_registry_mutex;
static std::vector<ThreadLocalBuffers*> g_tls_registry;


// true only after writer thread is fully stopped
static std::atomic<bool> g_trace_finalized{false};

// Active CUPTI callback counter
static std::atomic<uint64_t> g_active_callbacks{0};

//local buffer for api, nvtx record in each thread
struct ThreadLocalBuffers {

    std::mutex mtx;

    std::vector<ApiRecord> api_records;
    std::vector<NvtxRangeRecord> nvtx_records;

    ThreadLocalBuffers() {
        api_records.reserve(API_LOCAL_FLUSH_N);
        nvtx_records.reserve(NVTX_LOCAL_FLUSH_N);

        //pushing the local buffer's address into global register
        {
            std::lock_guard<std::mutex> lock(g_tls_registry_mutex);
            g_tls_registry.push_back(this);
        } // unlocked here

        fprintf(stderr, "[TLS REGISTER] buffer=%p\n", (void*)this);
        fflush(stderr);
    }

    ~ThreadLocalBuffers() {
        // This runs when this specific thread exits.
        // It flushes this thread's remaining local records.
        //
        // Do NOT call CUPTI here.
        // Do NOT write SQLite here.
        // Do NOT fprintf heavily here.

        fprintf(stderr, "[TLS DESTRUCTOR] buffer=%p\n", (void*)this);
        fflush(stderr);

        std::vector<ApiRecord> api_tmp;
        std::vector<NvtxRangeRecord> nvtx_tmp;

        try {   
            {
                // IMPORTANT:
                // registry lock first,
                // local buffer lock second
                std::lock_guard<std::mutex> reg_lock(g_tls_registry_mutex);
                std::lock_guard<std::mutex> local_lock(mtx);

                // move leftover records into tmp
                api_tmp.swap(api_records);
                nvtx_tmp.swap(nvtx_records);

                // remove my address from the registry
                auto it = std::find(
                    g_tls_registry.begin(),
                    g_tls_registry.end(),
                    this
                );

                if (it != g_tls_registry.end()) {
                    g_tls_registry.erase(it);
                }
            }

            // 3. Flush to the global queues after releasing the locks.
            // If the writer has already fully stopped, nobody is left to write them.
            if (!g_trace_finalized.load(std::memory_order_acquire)) {
                if (!api_tmp.empty()) {
                    flushApiVectorToGlobal(api_tmp);
                }

                if (!nvtx_tmp.empty()) {
                    flushNvtxVectorToGlobal(nvtx_tmp);
                }
            } 
            else {
                if (!api_tmp.empty() || !nvtx_tmp.empty()) {
                    fprintf(stderr,
                            "[TLS DESTRUCTOR WARNING] trace finalized; dropping leftover api=%zu nvtx=%zu buffer=%p\n",
                            api_tmp.size(),
                            nvtx_tmp.size(),
                            (void*)this);
                    fflush(stderr);
                }
            }
            fprintf(stderr,
                "[TLS DESTRUCTOR DONE] buffer=%p flushed_api=%zu flushed_nvtx=%zu\n",
                (void*)this,
                api_tmp.size(),
                nvtx_tmp.size());
            fflush(stderr);

        } catch (...) {
            // Never throw from destructor.
        }
    }
};

static thread_local ThreadLocalBuffers t_buffers;


//mutex to lock 
static std::mutex g_api_mutex;
static std::mutex g_nvtx_mutex;


static std::vector<ApiRecord> g_api_queue;
static std::vector<NvtxRangeRecord> g_nvtx_queue;


static std::atomic<uint64_t> g_dropped_api{0};
static std::atomic<uint64_t> g_dropped_nvtx{0};

static std::atomic<uint64_t> g_api_id{1};
static std::atomic<uint64_t> g_nvtx_id{1};

static thread_local std::vector<NvtxFrame> g_nvtx_stack;

static thread_local std::unordered_map<uint64_t, ApiEnterInfo> g_inflight_api;


static CUpti_SubscriberHandle g_subscriber;

static std::thread* g_writer_thread = nullptr;

static std::atomic<bool> g_writer_running{false};
static std::atomic<bool> g_tracing_enabled{false};



static std::atomic<uint64_t> g_writer_api_total{0};
static std::atomic<uint64_t> g_writer_nvtx_total{0};



static constexpr int WRITER_POLL_MS = 50;

//Helper function
static uint64_t getThreadId() {
    return static_cast<uint64_t>(syscall(SYS_gettid));
}

static void flushApiVectorToGlobal(std::vector<ApiRecord>& local) {
    if (local.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_api_mutex);

    if (g_api_queue.size() + local.size() <= API_GLOBAL_MAX) {
        g_api_queue.insert(
            g_api_queue.end(),
            local.begin(),
            local.end()
        );
    } else {
        g_dropped_api += local.size();
    }

    local.clear();
}

static void flushNvtxVectorToGlobal(std::vector<NvtxRangeRecord>& local){
    if (local.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_nvtx_mutex);

    if (g_nvtx_queue.size() + local.size() <= NVTX_GLOBAL_MAX){
        g_nvtx_queue.insert(
            g_nvtx_queue.end(),
            local.begin(),
            local.end()
        );
    } else {
        g_dropped_nvtx += local.size();
    }

    local.clear();
}


static void flushApiLocalBuffer(){
    std::vector<ApiRecord> api_tmp;

    {
        std::lock_guard<std::mutex> local_lock(t_buffers.mtx);
        api_tmp.swap(t_buffers.api_records);
    }
    if (!api_tmp.empty()) {

        flushApiVectorToGlobal(api_tmp);
    }
}


static void pushApiRecord(const ApiRecord& rec){
    std::vector<ApiRecord> api_tmp;

    {
        std::lock_guard<std::mutex> local_lock(t_buffers.mtx);

        t_buffers.api_records.push_back(rec);

        if (t_buffers.api_records.size() >= API_LOCAL_FLUSH_N) {
            api_tmp.swap(t_buffers.api_records);
        }
    }

    if (!api_tmp.empty()) {
        flushApiVectorToGlobal(api_tmp);
    }
}

static void flushNvtxLocalBuffer(){
    std::vector<NvtxRangeRecord> nvtx_tmp;

    {
        std::lock_guard<std::mutex> local_lock(t_buffers.mtx);
        nvtx_tmp.swap(t_buffers.nvtx_records);
    }

    if (!nvtx_tmp.empty()) {
        flushNvtxVectorToGlobal(nvtx_tmp);
    }
}

static void pushNvtxRecord(const NvtxRangeRecord& rec){
    std::vector<NvtxRangeRecord> nvtx_tmp;

    {
        std::lock_guard<std::mutex> local_lock(t_buffers.mtx);

        t_buffers.nvtx_records.push_back(rec);

        if (t_buffers.nvtx_records.size() >= NVTX_LOCAL_FLUSH_N) {
            nvtx_tmp.swap(t_buffers.nvtx_records);
        }
    }

    if (!nvtx_tmp.empty()) {
        flushNvtxVectorToGlobal(nvtx_tmp);
    }
}

static void flushAllThreadLocalBuffers() {

    std::vector<std::vector<ApiRecord>> api_batches;
    std::vector<std::vector<NvtxRangeRecord>> nvtx_batches;

    {
        std::lock_guard<std::mutex> reg_lock(g_tls_registry_mutex);

        fprintf(stderr, "[TLS FLUSH ALL] registry_size=%zu\n", g_tls_registry.size());
        fflush(stderr);

        for (ThreadLocalBuffers* buf : g_tls_registry) {
            if (buf == nullptr) {
                continue;
            }

            std::vector<ApiRecord> api_tmp;
            std::vector<NvtxRangeRecord> nvtx_tmp;

            {
                std::lock_guard<std::mutex> local_lock(buf->mtx);

                api_tmp.swap(buf->api_records);
                nvtx_tmp.swap(buf->nvtx_records);
            }

            if (!api_tmp.empty()) {
                api_batches.push_back(std::move(api_tmp));
            }

            if (!nvtx_tmp.empty()) {
                nvtx_batches.push_back(std::move(nvtx_tmp));
            }
        }
    }

    for (auto& batch : api_batches) {
        flushApiVectorToGlobal(batch);
    }

    for (auto& batch : nvtx_batches) {
        flushNvtxVectorToGlobal(batch);
    }

    fprintf(stderr,
            "[TLS FLUSH ALL DONE] api_batches=%zu nvtx_batches=%zu\n",
            api_batches.size(),
            nvtx_batches.size());
    fflush(stderr);
}

static unsigned long long currentNvtxId(){

    if (g_nvtx_stack.empty()) {
        return 0;  // 0 means NO_NVTX
    }
    return g_nvtx_stack.back().id;
}


static std::string currentNvtxIdPath(){

    if (g_nvtx_stack.empty()) {
        return "0";
    }

    std::string path;

    for (size_t i = 0; i < g_nvtx_stack.size(); i++) {
        if (i > 0) path += "/";
        path += std::to_string(g_nvtx_stack[i].id);
    }

    return path;
}


static const char* domainName(CUpti_CallbackDomain domain) {
    switch (domain) {
        case CUPTI_CB_DOMAIN_RUNTIME_API:
            return "RUNTIME";
        case CUPTI_CB_DOMAIN_DRIVER_API:
            return "DRIVER";
        default:
            return "OTHER";
    }
}

static const char* domainToString(uint32_t domain) {
    switch (domain) {
        case CUPTI_CB_DOMAIN_RUNTIME_API:
            return "RUNTIME";
        case CUPTI_CB_DOMAIN_DRIVER_API:
            return "DRIVER";
        case CUPTI_CB_DOMAIN_NVTX:
            return "NVTX";
        default:
            return "UNKNOWN";
    }
}

static bool isInterestingApi(const char *name) {
    if (name == nullptr) return false;

    return strstr(name, "LaunchKernel") ||
           strstr(name, "Memcpy") ||
           strstr(name, "Memset") ||
           strstr(name, "Synchronize");
}

static bool isMemoryMgmtApi(const char* f) {

    return strstr(f, "Malloc")    ||   // cudaMalloc, cudaMallocAsync, cuMemAllocHost...
           strstr(f, "Free"); // cudaFree, cudaFreeAsync, cuMemFree...
}


static std::string currentNvtxName() {
    if (g_nvtx_stack.empty()) {
        return "NO_NVTX";
    }
    return g_nvtx_stack.back().name;
}


static void handleNvtxCallback(
    CUpti_CallbackId cbid,
    const void *cbdata
) {

    const CUpti_NvtxData *nvtxData = reinterpret_cast<const CUpti_NvtxData *>(cbdata);

    const char *fname = nvtxData->functionName ? nvtxData->functionName : "UNKNOWN_NVTX_FUNC";

    const uint64_t current_thread_id = getThreadId();

    //-----------------------------------------------
    // NVTX PUSH
    //-----------------------------------------------
    if (cbid == CUPTI_CBID_NVTX_nvtxRangePushA) {

        const nvtxRangePushA_params *params = reinterpret_cast<const nvtxRangePushA_params *>(nvtxData->functionParams);

        const char *msg = params && params->message ? params->message : "UNKNOWN_NVTX";

        uint64_t ts_ns = 0;
        cuptiGetTimestamp(&ts_ns);

        NvtxFrame frame;
        frame.id = g_nvtx_id.fetch_add(1);
        frame.parent_id = g_nvtx_stack.empty() ? 0 : g_nvtx_stack.back().id;

        frame.depth = static_cast<uint32_t>(g_nvtx_stack.size() + 1);
        frame.thread_id = getThreadId();
        frame.push_ns = ts_ns;
        frame.name = msg;

        g_nvtx_stack.push_back(frame);


            
        TRACE_LOG(
            "[NVTX PUSH] nvtx_id=%llu parent_nvtx_id=%llu name=%s depth=%u\n",
            (unsigned long long)frame.id,
            (unsigned long long)frame.parent_id,
            frame.name.c_str(),
            frame.depth);
    }

    //-----------------------------------------------
    // NVTX POP UP
    //-----------------------------------------------
    else if (cbid == CUPTI_CBID_NVTX_nvtxRangePop) {

        uint64_t pop_ns = 0;
        cuptiGetTimestamp(&pop_ns);


        if (!g_nvtx_stack.empty()) {

            NvtxFrame frame = g_nvtx_stack.back();
            g_nvtx_stack.pop_back();

            NvtxRangeRecord rec;
            rec.nvtx_id = frame.id;
            rec.parent_id = frame.parent_id;
            rec.depth = frame.depth;
            rec.thread_id = frame.thread_id;
            rec.push_ns = frame.push_ns;
            rec.pop_ns = pop_ns;
            rec.name = frame.name;

            pushNvtxRecord(rec);



            TRACE_LOG(
                    "[NVTX POP ] nvtx_id=%llu parent_nvtx_id=%llu name=%s depth=%u duration_us=%.3f\n",
                    (unsigned long long)rec.nvtx_id,
                    (unsigned long long)rec.parent_id,
                    rec.name.c_str(),
                    rec.depth,
                    (rec.pop_ns - rec.push_ns) / 1000.0);

        } 
        else {
            fprintf(stderr, "[NVTX POP ] empty stack func=%s\n", fname);
            fflush(stderr);
        }
    }
}

struct CallbackGuard {
    CallbackGuard() {
        g_active_callbacks.fetch_add(1, std::memory_order_acq_rel);
    }

    ~CallbackGuard() {
        g_active_callbacks.fetch_sub(1, std::memory_order_acq_rel);
    }
};

void CUPTIAPI callbackFunc(
    void *userdata,
    CUpti_CallbackDomain domain,
    CUpti_CallbackId cbid,
    const void *cbdata
) {

    CallbackGuard callback_guard;


    if (!g_tracing_enabled.load(std::memory_order_acquire)) {
        return;
    }


    //-----------------------------------------------
    // NVTX 
    //-----------------------------------------------
    if (domain == CUPTI_CB_DOMAIN_NVTX) {
        handleNvtxCallback(cbid, cbdata);
        return;
    }
    
    if (domain != CUPTI_CB_DOMAIN_RUNTIME_API &&
        domain != CUPTI_CB_DOMAIN_DRIVER_API) {
        return;
    }

    const CUpti_CallbackData *data = reinterpret_cast<const CUpti_CallbackData *>(cbdata);

    const char *fname = data->functionName ? data->functionName : "UNKNOWN";

    if (!isInterestingApi(fname) && !isMemoryMgmtApi(fname )) {
        return;
    }

    //-----------------------------------------------
    // API ENTER
    //-----------------------------------------------
    if (data->callbackSite == CUPTI_API_ENTER) {

        uint64_t my_api_id = g_api_id.fetch_add(1);


        unsigned long long active_nvtx_id = currentNvtxId();

        //std::string nvtx_id_path = currentNvtxIdPath();

        if (data->correlationData != nullptr) {
            *(data->correlationData) = my_api_id;
        }

        uint64_t enter_ns = 0;
        cuptiGetTimestamp(&enter_ns);


        g_inflight_api[my_api_id] = ApiEnterInfo{
            my_api_id,
            active_nvtx_id,
            enter_ns
        };

        TRACE_LOG(
            "[API ENTER] api_id=%llu corr=%u func=%s active_nvtx_id=%llu\n",
            (unsigned long long)my_api_id,
            data->correlationId,
            fname,
            (unsigned long long)active_nvtx_id);
    }
    //-----------------------------------------------
    // API EXIT
    //-----------------------------------------------
    else if (data->callbackSite == CUPTI_API_EXIT) {
        uint64_t my_api_id = 0;

        if (data->correlationData != nullptr) {
            my_api_id = *(data->correlationData);
        }

        uint64_t exit_ns = 0;
        cuptiGetTimestamp(&exit_ns);

        //unsigned long long active_nvtx_id = 0;
        //std::string nvtx_id_path = "0";

        auto it = g_inflight_api.find(my_api_id);

        if (it != g_inflight_api.end()) {

            uint64_t enter_ns = it->second.enter_ns;
            uint64_t duration_ns = 0;

            if (exit_ns >= enter_ns) {
                duration_ns = exit_ns - enter_ns;
            }

            // active_nvtx_id = it->second.active_nvtx_id;
            // nvtx_id_path = it->second.nvtx_id_path;
            ApiRecord rec;
            rec.api_id = my_api_id;
            rec.correlation_id = data->correlationId;
            rec.domain = static_cast<uint32_t>(domain);
            rec.cbid = static_cast<uint32_t>(cbid);
            rec.active_nvtx_id = it->second.active_nvtx_id;
            rec.thread_id = getThreadId();
            rec.enter_ns = enter_ns;
            rec.exit_ns = exit_ns;
            rec.duration_ns = duration_ns;
            rec.func_name = fname;


            pushApiRecord(rec);

            TRACE_LOG(
                "[API EXIT ] api_id=%llu corr=%u func=%s active_nvtx_id=%llu duration_us=%.3f\n",
                (unsigned long long)rec.api_id,
                rec.correlation_id,
                rec.func_name,
                (unsigned long long)rec.active_nvtx_id,
                rec.duration_ns / 1000.0);


            g_inflight_api.erase(it);
        }

        
    }
}

static void createTables(sqlite3* db) {
    
    const char* sql = R"SQL(
        CREATE TABLE IF NOT EXISTS api_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            api_id INTEGER,
            correlation_id INTEGER,
            domain TEXT,
            cbid INTEGER,
            active_nvtx_id INTEGER,
            thread_id INTEGER,
            enter_ns INTEGER,
            exit_ns INTEGER,
            duration_ns INTEGER,
            func_name TEXT
        );

        CREATE TABLE IF NOT EXISTS nvtx_ranges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nvtx_id INTEGER,
            parent_id INTEGER,
            depth INTEGER,
            thread_id INTEGER,
            push_ns INTEGER,
            pop_ns INTEGER,
            duration_ns INTEGER,
            name TEXT
        );
        CREATE TABLE IF NOT EXISTS kernel_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            name TEXT,

            correlation_id INTEGER,
            stream_id INTEGER,
            device_id INTEGER,
            context_id INTEGER,

            start_ns INTEGER,
            end_ns INTEGER,
            duration_ns INTEGER,
            completed_ns INTEGER,

            grid_x INTEGER,
            grid_y INTEGER,
            grid_z INTEGER,

            block_x INTEGER,
            block_y INTEGER,
            block_z INTEGER
        );

        CREATE TABLE IF NOT EXISTS memcpy_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            copy_kind TEXT,
            bytes INTEGER,

            correlation_id INTEGER,
            runtime_correlation_id INTEGER,

            stream_id INTEGER,
            device_id INTEGER,
            context_id INTEGER,

            start_ns INTEGER,
            end_ns INTEGER,
            duration_ns INTEGER
        );

        CREATE INDEX IF NOT EXISTS idx_api_correlation
            ON api_records(correlation_id);

        CREATE INDEX IF NOT EXISTS idx_kernel_correlation
            ON kernel_records(correlation_id);

        CREATE INDEX IF NOT EXISTS idx_memcpy_correlation
            ON memcpy_records(correlation_id);

        CREATE INDEX IF NOT EXISTS idx_memcpy_runtime_correlation
            ON memcpy_records(runtime_correlation_id);

        CREATE INDEX IF NOT EXISTS idx_api_nvtx
            ON api_records(active_nvtx_id);

        CREATE INDEX IF NOT EXISTS idx_kernel_stream_time
            ON kernel_records(device_id, context_id, stream_id, start_ns);

        CREATE INDEX IF NOT EXISTS idx_memcpy_stream_time
            ON memcpy_records(device_id, context_id, stream_id, start_ns);

    )SQL";

    char* err = nullptr;

    int rc = sqlite3_exec(db, sql, nullptr, nullptr, &err);

    if (rc != SQLITE_OK) {
        fprintf(stderr, "[SQLite] createTables error: %s\n", err ? err : "unknown");
        sqlite3_free(err);
    }
}

static void insertNvtxBatch(sqlite3* db, const std::vector<NvtxRangeRecord>& batch) {
    if (batch.empty()) return;

    sqlite3_exec(db, "BEGIN TRANSACTION;", nullptr, nullptr, nullptr);

    const char* sql = R"SQL(
        INSERT INTO nvtx_ranges (
            nvtx_id,
            parent_id,
            depth,
            thread_id,
            push_ns,
            pop_ns,
            duration_ns,
            name
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
    )SQL";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);

    if (rc != SQLITE_OK) {
        fprintf(stderr, "[SQLite] prepare nvtx insert failed: %s\n", sqlite3_errmsg(db));
        sqlite3_exec(db, "ROLLBACK;", nullptr, nullptr, nullptr);
        return;
    }

    for (const auto& rec : batch) {
        uint64_t duration_ns = 0;
        if (rec.pop_ns >= rec.push_ns) {
            duration_ns = rec.pop_ns - rec.push_ns;
        }

        sqlite3_bind_int64(stmt, 1, rec.nvtx_id);
        sqlite3_bind_int64(stmt, 2, rec.parent_id);
        sqlite3_bind_int64(stmt, 3, rec.depth);
        sqlite3_bind_int64(stmt, 4, rec.thread_id);
        sqlite3_bind_int64(stmt, 5, rec.push_ns);
        sqlite3_bind_int64(stmt, 6, rec.pop_ns);
        sqlite3_bind_int64(stmt, 7, duration_ns);
        sqlite3_bind_text(stmt, 8, rec.name.c_str(), -1, SQLITE_TRANSIENT);

        rc = sqlite3_step(stmt);

        if (rc != SQLITE_DONE) {
            fprintf(stderr, "[SQLite] nvtx insert failed: %s\n", sqlite3_errmsg(db));
        }

        sqlite3_reset(stmt);
        sqlite3_clear_bindings(stmt);
    }

    sqlite3_finalize(stmt);
    sqlite3_exec(db, "COMMIT;", nullptr, nullptr, nullptr);
}

static void insertApiBatch(sqlite3* db, const std::vector<ApiRecord>& batch) {
    if (batch.empty()) return;

    sqlite3_exec(db, "BEGIN TRANSACTION;", nullptr, nullptr, nullptr);

    const char* sql = R"SQL(
        INSERT INTO api_records (
            api_id,
            correlation_id,
            domain,
            cbid,
            active_nvtx_id,
            thread_id,
            enter_ns,
            exit_ns,
            duration_ns,
            func_name
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    )SQL";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);

    if (rc != SQLITE_OK) {
        fprintf(stderr, "[SQLite] prepare api insert failed: %s\n", sqlite3_errmsg(db));
        sqlite3_exec(db, "ROLLBACK;", nullptr, nullptr, nullptr);
        return;
    }

    for (const auto& rec : batch) {
        sqlite3_bind_int64(stmt, 1, rec.api_id);
        sqlite3_bind_int64(stmt, 2, rec.correlation_id);
        sqlite3_bind_text(stmt, 3, domainToString(rec.domain), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int64(stmt, 4, rec.cbid);
        sqlite3_bind_int64(stmt, 5, rec.active_nvtx_id);
        sqlite3_bind_int64(stmt, 6, rec.thread_id);
        sqlite3_bind_int64(stmt, 7, rec.enter_ns);
        sqlite3_bind_int64(stmt, 8, rec.exit_ns);
        sqlite3_bind_int64(stmt, 9, rec.duration_ns);
        sqlite3_bind_text(stmt, 10, rec.func_name ? rec.func_name : "", -1, SQLITE_TRANSIENT);

        rc = sqlite3_step(stmt);

        if (rc != SQLITE_DONE) {
            fprintf(stderr, "[SQLite] api insert failed: %s\n", sqlite3_errmsg(db));
        }

        sqlite3_reset(stmt);
        sqlite3_clear_bindings(stmt);
    }

    sqlite3_finalize(stmt);
    sqlite3_exec(db, "COMMIT;", nullptr, nullptr, nullptr);
}

static void insertKernelBatch(
    sqlite3* db,
    const std::vector<KernelActivityRecord>& batch
) {
    if (batch.empty()) {
        return;
    }

    int rc = sqlite3_exec(db, "BEGIN TRANSACTION;", nullptr, nullptr, nullptr);

    if (rc != SQLITE_OK) {
        fprintf(stderr, "[SQLite] begin kernel transaction failed: %s\n", sqlite3_errmsg(db));
        fflush(stderr);
        return;
    }

    const char* sql = R"SQL(
        INSERT INTO kernel_records (
            name,
            correlation_id,
            stream_id,
            device_id,
            context_id,
            start_ns,
            end_ns,
            duration_ns,
            completed_ns,
            grid_x,
            grid_y,
            grid_z,
            block_x,
            block_y,
            block_z
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    )SQL";

    sqlite3_stmt* stmt = nullptr;

    rc = sqlite3_prepare_v2( db, sql, -1, &stmt, nullptr);

    if (rc != SQLITE_OK) {
        fprintf(stderr,
                "[SQLite] prepare kernel insert failed: %s\n",
                sqlite3_errmsg(db));
        fflush(stderr);

        sqlite3_exec(db,"ROLLBACK;", nullptr, nullptr, nullptr);

        return;
    }

    bool success = true;

    for (const auto& rec : batch) {

        sqlite3_bind_text(
            stmt,
            1,
            rec.name.c_str(),
            -1,
            SQLITE_TRANSIENT
        );

        sqlite3_bind_int64(
            stmt,
            2,
            static_cast<sqlite3_int64>(
                rec.correlation_id
            )
        );

        sqlite3_bind_int64(
            stmt,
            3,
            static_cast<sqlite3_int64>(
                rec.stream_id
            )
        );

        sqlite3_bind_int64(
            stmt,
            4,
            static_cast<sqlite3_int64>(
                rec.device_id
            )
        );

        sqlite3_bind_int64(
            stmt,
            5,
            static_cast<sqlite3_int64>(
                rec.context_id
            )
        );

        sqlite3_bind_int64(
            stmt,
            6,
            static_cast<sqlite3_int64>(
                rec.start_ns
            )
        );

        sqlite3_bind_int64(
            stmt,
            7,
            static_cast<sqlite3_int64>(
                rec.end_ns
            )
        );

        sqlite3_bind_int64(
            stmt,
            8,
            static_cast<sqlite3_int64>(
                rec.duration_ns
            )
        );

        sqlite3_bind_int64(
            stmt,
            9,
            static_cast<sqlite3_int64>(
                rec.completed_ns
            )
        );

        sqlite3_bind_int(
            stmt,
            10,
            rec.grid_x
        );

        sqlite3_bind_int(
            stmt,
            11,
            rec.grid_y
        );

        sqlite3_bind_int(
            stmt,
            12,
            rec.grid_z
        );

        sqlite3_bind_int(
            stmt,
            13,
            rec.block_x
        );

        sqlite3_bind_int(
            stmt,
            14,
            rec.block_y
        );

        sqlite3_bind_int(
            stmt,
            15,
            rec.block_z
        );

        rc = sqlite3_step(stmt);

        if (rc != SQLITE_DONE) {
            fprintf(stderr,"[SQLite] kernel insert failed: %s\n",sqlite3_errmsg(db));
            fflush(stderr);

            success = false;
            break;
        }

        // reuse the same prepared statement for the next record
        sqlite3_reset(stmt);
        sqlite3_clear_bindings(stmt);
    }

    sqlite3_finalize(stmt);

    if (success) {
        rc = sqlite3_exec(db, "COMMIT;", nullptr, nullptr, nullptr);

        if (rc != SQLITE_OK) {
            fprintf(stderr,
                    "[SQLite] kernel commit failed: %s\n",
                    sqlite3_errmsg(db));
            fflush(stderr);
        }
    } else {
        sqlite3_exec(db,"ROLLBACK;",nullptr,nullptr,nullptr);
    }
}
static void insertMemcpyBatch(
    sqlite3* db,
    const std::vector<MemcpyActivityRecord>& batch
) {
    if (batch.empty()) {
        return;
    }

    int rc = sqlite3_exec(
        db,
        "BEGIN TRANSACTION;",
        nullptr,
        nullptr,
        nullptr
    );

    if (rc != SQLITE_OK) {
        fprintf(stderr,
                "[SQLite] begin memcpy transaction failed: %s\n",
                sqlite3_errmsg(db));
        fflush(stderr);
        return;
    }

    const char* sql = R"SQL(
        INSERT INTO memcpy_records (
            copy_kind,
            bytes,
            correlation_id,
            runtime_correlation_id,
            stream_id,
            device_id,
            context_id,
            start_ns,
            end_ns,
            duration_ns
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    )SQL";

    sqlite3_stmt* stmt = nullptr;

    rc = sqlite3_prepare_v2(
        db,
        sql,
        -1,
        &stmt,
        nullptr
    );

    if (rc != SQLITE_OK) {
        fprintf(stderr,
                "[SQLite] prepare memcpy insert failed: %s\n",
                sqlite3_errmsg(db));
        fflush(stderr);

        sqlite3_exec(
            db,
            "ROLLBACK;",
            nullptr,
            nullptr,
            nullptr
        );

        return;
    }

    bool success = true;

    for (const auto& rec : batch) {

        // uint8_t copy_kind → human-readable string
        sqlite3_bind_text(
            stmt,
            1,
            memcpyKindToString(rec.copy_kind),
            -1,
            SQLITE_TRANSIENT
        );

        sqlite3_bind_int64(
            stmt,
            2,
            static_cast<sqlite3_int64>(
                rec.bytes
            )
        );

        sqlite3_bind_int64(
            stmt,
            3,
            static_cast<sqlite3_int64>(
                rec.correlation_id
            )
        );

        sqlite3_bind_int64(
            stmt,
            4,
            static_cast<sqlite3_int64>(
                rec.runtime_correlation_id
            )
        );

        sqlite3_bind_int64(
            stmt,
            5,
            static_cast<sqlite3_int64>(
                rec.stream_id
            )
        );

        sqlite3_bind_int64(
            stmt,
            6,
            static_cast<sqlite3_int64>(
                rec.device_id
            )
        );

        sqlite3_bind_int64(
            stmt,
            7,
            static_cast<sqlite3_int64>(
                rec.context_id
            )
        );

        sqlite3_bind_int64(
            stmt,
            8,
            static_cast<sqlite3_int64>(
                rec.start_ns
            )
        );

        sqlite3_bind_int64(
            stmt,
            9,
            static_cast<sqlite3_int64>(
                rec.end_ns
            )
        );

        sqlite3_bind_int64(
            stmt,
            10,
            static_cast<sqlite3_int64>(
                rec.duration_ns
            )
        );

        rc = sqlite3_step(stmt);

        if (rc != SQLITE_DONE) {
            fprintf(stderr,
                    "[SQLite] memcpy insert failed: %s\n",
                    sqlite3_errmsg(db));
            fflush(stderr);

            success = false;
            break;
        }

        sqlite3_reset(stmt);
        sqlite3_clear_bindings(stmt);
    }

    sqlite3_finalize(stmt);

    if (success) {
        rc = sqlite3_exec(
            db,
            "COMMIT;",
            nullptr,
            nullptr,
            nullptr
        );

        if (rc != SQLITE_OK) {
            fprintf(stderr,
                    "[SQLite] memcpy commit failed: %s\n",
                    sqlite3_errmsg(db));
            fflush(stderr);
        }
    } else {
        sqlite3_exec(
            db,
            "ROLLBACK;",
            nullptr,
            nullptr,
            nullptr
        );
    }
}


static void writerThreadMain() {
    fprintf(stderr, "[Writer] Writer thread started\n");
    fflush(stderr);

    sqlite3* db = nullptr;
    const std::string db_path = dbPathForRank();

    fprintf(stderr, "[Writer] opening DB: %s\n", db_path.c_str());
    fflush(stderr);

    int rc = sqlite3_open(db_path.c_str(), &db);
    
    if (rc != SQLITE_OK) {
        fprintf(stderr,
                "[SQLite] cannot open database %s: %s\n",
                db_path.c_str(),
                db ? sqlite3_errmsg(db) : "db is null");
        fflush(stderr);

        if (db) {
            sqlite3_close(db);
        }

        fprintf(stderr, "[Writer] Writer thread stopped because SQLite open failed\n");
        fflush(stderr);
        return;
    }

    createTables(db);

    sqlite3_exec(db, "PRAGMA journal_mode=WAL;", nullptr, nullptr, nullptr);
    sqlite3_exec(db, "PRAGMA synchronous=NORMAL;", nullptr, nullptr, nullptr);
    sqlite3_exec(db, "PRAGMA temp_store=MEMORY;", nullptr, nullptr, nullptr);
    sqlite3_busy_timeout(db, 5000);


    while (g_writer_running.load(std::memory_order_acquire)) {
        std::vector<ApiRecord> writer_api_buffer;
        std::vector<NvtxRangeRecord> writer_nvtx_buffer;


        std::vector<KernelActivityRecord> writer_kernel_buffer;
        std::vector<MemcpyActivityRecord> writer_memcpy_buffer;

        // 1. Drain API queue into writer-local buffer
        {
            std::lock_guard<std::mutex> lock(g_api_mutex);

            if (!g_api_queue.empty()) {
                writer_api_buffer.swap(g_api_queue);
            }
        }

        // 2. Drain NVTX queue into writer-local buffer
        {
            std::lock_guard<std::mutex> lock(g_nvtx_mutex);

            if (!g_nvtx_queue.empty()) {
                writer_nvtx_buffer.swap(g_nvtx_queue);
            }
        }
        // 3. Drain Activity queues into writer-local buffers
        {
            std::lock_guard<std::mutex> lock(g_activity_mutex);

            if (!g_kernel_activity_queue.empty()) {
                writer_kernel_buffer.swap(
                    g_kernel_activity_queue
                );
            }

            if (!g_memcpy_activity_queue.empty()) {
                writer_memcpy_buffer.swap(
                    g_memcpy_activity_queue
                );
            }
        }
        const bool has_records = 
        !writer_api_buffer.empty() ||
        !writer_nvtx_buffer.empty() ||
        !writer_kernel_buffer.empty() ||
        !writer_memcpy_buffer.empty();

        if (!has_records) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(WRITER_POLL_MS)
            );

            continue;
        }
        // ==========================================================
        // Write writer-local buffers to SQLite
        // ==========================================================

        if (!writer_api_buffer.empty()) {
            insertApiBatch(
                db,
                writer_api_buffer
            );
        }

        if (!writer_nvtx_buffer.empty()) {
            insertNvtxBatch(
                db,
                writer_nvtx_buffer
            );
        }

        if (!writer_kernel_buffer.empty()) {
            insertKernelBatch(
                db,
                writer_kernel_buffer
            );
        }

        if (!writer_memcpy_buffer.empty()) {
            insertMemcpyBatch(
                db,
                writer_memcpy_buffer
            );
        }

        // ==========================================================
        // Update writer totals
        // ==========================================================

        g_writer_api_total.fetch_add(
            writer_api_buffer.size(),
            std::memory_order_relaxed
        );

        g_writer_nvtx_total.fetch_add(
            writer_nvtx_buffer.size(),
            std::memory_order_relaxed
        );

        g_writer_kernel_total.fetch_add(
            writer_kernel_buffer.size(),
            std::memory_order_relaxed
        );

        g_writer_memcpy_total.fetch_add(
            writer_memcpy_buffer.size(),
            std::memory_order_relaxed
        );

        // ==========================================================
        // Debug output
        // ==========================================================

        TRACE_LOG(
            "[Writer] wrote "
            "api=%zu nvtx=%zu kernel=%zu memcpy=%zu | "
            "total api=%llu nvtx=%llu kernel=%llu memcpy=%llu\n",

            writer_api_buffer.size(),
            writer_nvtx_buffer.size(),
            writer_kernel_buffer.size(),
            writer_memcpy_buffer.size(),

            static_cast<unsigned long long>(
                g_writer_api_total.load(std::memory_order_relaxed)
            ),

            static_cast<unsigned long long>(
                g_writer_nvtx_total.load(std::memory_order_relaxed)
            ),

            static_cast<unsigned long long>(
                g_writer_kernel_total.load(std::memory_order_relaxed)
            ),

            static_cast<unsigned long long>(
                g_writer_memcpy_total.load(std::memory_order_relaxed)
            )
        );
        fflush(stderr);
    }

    // Final drain after stop signal
    // Final drain after stop signal
    while (true) {
        std::vector<ApiRecord> writer_api_buffer;
        std::vector<NvtxRangeRecord> writer_nvtx_buffer;
        std::vector<KernelActivityRecord> writer_kernel_buffer;
        std::vector<MemcpyActivityRecord> writer_memcpy_buffer;

        // ----------------------------------------------------------
        // Drain API queue
        // ----------------------------------------------------------
        {
            std::lock_guard<std::mutex> lock(g_api_mutex);

            if (!g_api_queue.empty()) {
                writer_api_buffer.swap(g_api_queue);
            }
        }

        // ----------------------------------------------------------
        // Drain NVTX queue
        // ----------------------------------------------------------
        {
            std::lock_guard<std::mutex> lock(g_nvtx_mutex);

            if (!g_nvtx_queue.empty()) {
                writer_nvtx_buffer.swap(g_nvtx_queue);
            }
        }

        // ----------------------------------------------------------
        // Drain Activity queues
        // ----------------------------------------------------------
        {
            std::lock_guard<std::mutex> lock(g_activity_mutex);

            if (!g_kernel_activity_queue.empty()) {
                writer_kernel_buffer.swap(
                    g_kernel_activity_queue
                );
            }

            if (!g_memcpy_activity_queue.empty()) {
                writer_memcpy_buffer.swap(
                    g_memcpy_activity_queue
                );
            }
        }

        // end the final drain once every global queue is empty
        const bool all_empty =
            writer_api_buffer.empty() &&
            writer_nvtx_buffer.empty() &&
            writer_kernel_buffer.empty() &&
            writer_memcpy_buffer.empty();

        if (all_empty) {
            break;
        }

        // ----------------------------------------------------------
        // Write all writer-local buffers to SQLite
        // ----------------------------------------------------------

        if (!writer_api_buffer.empty()) {
            insertApiBatch(
                db,
                writer_api_buffer
            );
        }

        if (!writer_nvtx_buffer.empty()) {
            insertNvtxBatch(
                db,
                writer_nvtx_buffer
            );
        }

        if (!writer_kernel_buffer.empty()) {
            insertKernelBatch(
                db,
                writer_kernel_buffer
            );
        }

        if (!writer_memcpy_buffer.empty()) {
            insertMemcpyBatch(
                db,
                writer_memcpy_buffer
            );
        }

        // ----------------------------------------------------------
        // Update totals
        // ----------------------------------------------------------

        g_writer_api_total.fetch_add(
            writer_api_buffer.size(),
            std::memory_order_relaxed
        );

        g_writer_nvtx_total.fetch_add(
            writer_nvtx_buffer.size(),
            std::memory_order_relaxed
        );

        g_writer_kernel_total.fetch_add(
            writer_kernel_buffer.size(),
            std::memory_order_relaxed
        );

        g_writer_memcpy_total.fetch_add(
            writer_memcpy_buffer.size(),
            std::memory_order_relaxed
        );

        // ----------------------------------------------------------
        // Debug output
        // ----------------------------------------------------------

        fprintf(
            stderr,
            "[Writer] final wrote "
            "api=%zu nvtx=%zu kernel=%zu memcpy=%zu | "
            "total api=%llu nvtx=%llu kernel=%llu memcpy=%llu\n",

            writer_api_buffer.size(),
            writer_nvtx_buffer.size(),
            writer_kernel_buffer.size(),
            writer_memcpy_buffer.size(),

            static_cast<unsigned long long>(
                g_writer_api_total.load(
                    std::memory_order_relaxed
                )
            ),

            static_cast<unsigned long long>(
                g_writer_nvtx_total.load(
                    std::memory_order_relaxed
                )
            ),

            static_cast<unsigned long long>(
                g_writer_kernel_total.load(
                    std::memory_order_relaxed
                )
            ),

            static_cast<unsigned long long>(
                g_writer_memcpy_total.load(
                    std::memory_order_relaxed
                )
            )
        );

        fflush(stderr);
    }

    sqlite3_close(db);

    fprintf(stderr, "[Writer] Writer thread stopped\n");
    fflush(stderr);
}


__attribute__((constructor))
static void initCuptiTracer() {


    fprintf(stderr, "[CUPTI] Initializing CUPTI callback tracer\n");
    fflush(stderr);

    g_api_queue.reserve(API_GLOBAL_MAX);
    g_nvtx_queue.reserve(NVTX_GLOBAL_MAX);

    g_writer_running.store(true, std::memory_order_release);
    g_tracing_enabled.store(true, std::memory_order_release);
    g_trace_finalized.store(false, std::memory_order_release);

    initActivityTracing();

    //g_writer_thread = std::thread(writerThreadMain);
    g_writer_thread = new std::thread(writerThreadMain);

    fprintf(stderr,
            "[CUPTI] constructor writer ptr=%p joinable=%d\n",
            (void*)g_writer_thread,
            g_writer_thread && g_writer_thread->joinable() ? 1 : 0);
    fflush(stderr);

    CHECK_CUPTI(cuptiSubscribe(
        &g_subscriber,
        (CUpti_CallbackFunc)callbackFunc,
        nullptr
    ));

    CHECK_CUPTI(cuptiEnableDomain(
        1,
        g_subscriber,
        CUPTI_CB_DOMAIN_RUNTIME_API
    ));

    CHECK_CUPTI(cuptiEnableDomain(
        1,
        g_subscriber,
        CUPTI_CB_DOMAIN_DRIVER_API
    ));

    CHECK_CUPTI(cuptiEnableDomain(
        1,
        g_subscriber,
        CUPTI_CB_DOMAIN_NVTX
    ));
    

    fprintf(stderr, "[CUPTI] Runtime + Driver API + NVTX callbacks enabled\n");
    fflush(stderr);

    clock_anchor_capture_init();

}

extern "C" void cupti_trace_stop() {


    clock_anchor_capture_finalize();


    fprintf(stderr, "[CUPTI] Explicit stop called\n");
    fflush(stderr);

    // 1. Stop accepting new callback records
    g_tracing_enabled.store(false, std::memory_order_release);

    // 2. Wait until callbacks already inside finish
    while (g_active_callbacks.load(std::memory_order_acquire) != 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

 
    // 3. Flush every registered thread-local buffer
    flushAllThreadLocalBuffers();

    fprintf(stderr, "[CUPTI Activity] flushing all activity records\n");
    fflush(stderr);

    cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);

    fprintf(stderr, "[CUPTI Activity] flush done\n");
    fflush(stderr);

    // Stop writer thread
    g_writer_running.store(false, std::memory_order_release);

    fprintf(stderr,
            "[CUPTI] BEFORE join writer ptr=%p joinable=%d\n",
            (void*)g_writer_thread,
            g_writer_thread && g_writer_thread->joinable() ? 1 : 0);
    fflush(stderr);


    if (g_writer_thread != nullptr && g_writer_thread->joinable()) {
        g_writer_thread->join();
    }

    g_trace_finalized.store(true, std::memory_order_release);

    fprintf(stderr,
            "[CUPTI] AFTER join writer ptr=%p joinable=%d\n",
            (void*)g_writer_thread,
            g_writer_thread && g_writer_thread->joinable() ? 1 : 0);
    fflush(stderr);

    fprintf(stderr,
            "[CUPTI] Writer totals: api=%llu nvtx=%llu dropped_api=%llu dropped_nvtx=%llu\n",
            (unsigned long long)g_writer_api_total.load(),
            (unsigned long long)g_writer_nvtx_total.load(),
            (unsigned long long)g_dropped_api.load(),
            (unsigned long long)g_dropped_nvtx.load());
    fflush(stderr);

    delete g_writer_thread;
    g_writer_thread = nullptr;

    clock_anchor_write_db(dbPathForRank().c_str());
}

__attribute__((destructor))
static void finiCuptiTracer() {

    fprintf(stderr, "[CUPTI] Finalizing CUPTI callback tracer\n");
    fflush(stderr);

    {
        std::lock_guard<std::mutex> lock(g_api_mutex);
        fprintf(stderr,
                "[CUPTI] Final API records buffered: %zu, dropped: %llu\n",
                g_api_queue.size(),
                (unsigned long long)g_dropped_api.load());
    }

    {
        std::lock_guard<std::mutex> lock(g_nvtx_mutex);
        fprintf(stderr,
                "[CUPTI] Final NVTX records buffered: %zu, dropped: %llu\n",
                g_nvtx_queue.size(),
                (unsigned long long)g_dropped_nvtx.load());
    }

    //cuptiUnsubscribe(g_subscriber);

}