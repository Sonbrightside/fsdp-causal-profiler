#include "clock_anchor.h"
#include <cupti.h>
#include <sqlite3.h>
#include <time.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>

// ---- clock anchor ----------------------------------------------------
// Hard constraint: must use the same clock id as Python's time.monotonic_ns().
// Linux CPython implements monotonic_ns with CLOCK_MONOTONIC, so this side
// must also use CLOCK_MONOTONIC (NOT RAW!).
static inline uint64_t host_ns_monotonic() {

    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

struct ClockAnchor {
    uint64_t cupti_ns;
    uint64_t host_ns;     // bracket midpoint
    uint64_t bracket_ns;  // h1 - h0 (uncertainty evidence, stored in the DB alongside)
    bool     valid      = false;
};


static ClockAnchor g_anchor_init;    
static ClockAnchor g_anchor_finalize;


static ClockAnchor capture_anchor(int n_tries = 64) {

    ClockAnchor best;
    best.bracket_ns = UINT64_MAX;

    for (int i = 0; i < n_tries; i++) {

        uint64_t h0 = host_ns_monotonic();
        uint64_t c  = 0;

        if (cuptiGetTimestamp(&c) != CUPTI_SUCCESS) continue;  // discard failed samples

        uint64_t h1 = host_ns_monotonic();
        uint64_t w  = h1 - h0;

        if (w < best.bracket_ns) {

            best.cupti_ns   = c;
            best.host_ns    = h0 + w / 2;
            best.bracket_ns = w;
            best.valid      = true;

        }
    }
    return best;
}

static void capture_and_log(const char* phase, ClockAnchor& slot) {

    slot = capture_anchor();

    // fail-fast (1): best of 64 tries exceeds 10µs, or all failed → anchor untrustworthy
    if (!slot.valid || slot.bracket_ns > 10000) {

        fprintf(stderr, "[ANCHOR][FATAL] phase=%s valid=%d bracket=%llu ns\n", phase, (int)slot.valid, (unsigned long long)slot.bracket_ns);
        fflush(stderr);
        abort();
    }

    // fail-fast (2): one-line grep verdict + backup in case the DB is lost
    fprintf(stderr, "[ANCHOR] phase=%s pid=%d cupti=%llu host=%llu bracket=%lluns\n",
            phase, (int)getpid(),
            (unsigned long long)slot.cupti_ns,
            (unsigned long long)slot.host_ns,
            (unsigned long long)slot.bracket_ns);
    fflush(stderr);
}

void clock_anchor_capture_init(){ 
    capture_and_log("init",     g_anchor_init); 
}

void clock_anchor_capture_finalize() { 
    capture_and_log("finalize", g_anchor_finalize); 
}

static void insert_one(sqlite3* db, const char* phase, const ClockAnchor& a) {

    sqlite3_stmt* stmt = nullptr;

    const char* sql =
        "INSERT INTO clock_anchors (phase, pid, cupti_ns, host_ns, bracket_ns) "
        "VALUES (?, ?, ?, ?, ?);";

    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        fprintf(stderr, "[ANCHOR] prepare failed: %s\n", sqlite3_errmsg(db));
        return;
    }
    sqlite3_bind_text (stmt, 1, phase, -1, SQLITE_TRANSIENT);
    sqlite3_bind_int  (stmt, 2, (int)getpid());
    sqlite3_bind_int64(stmt, 3, (sqlite3_int64)a.cupti_ns);
    sqlite3_bind_int64(stmt, 4, (sqlite3_int64)a.host_ns);
    sqlite3_bind_int64(stmt, 5, (sqlite3_int64)a.bracket_ns);

    if (sqlite3_step(stmt) != SQLITE_DONE) {
        fprintf(stderr, "[ANCHOR] insert failed: %s\n", sqlite3_errmsg(db));
    }

    sqlite3_finalize(stmt);
}

void clock_anchor_write_db(const char* db_path) {
    if (!g_anchor_init.valid && !g_anchor_finalize.valid) {
        fprintf(stderr, "[ANCHOR] nothing to write\n");
        return;
    }

    sqlite3* db = nullptr;
    if (sqlite3_open(db_path, &db) != SQLITE_OK) {
        fprintf(stderr, "[ANCHOR] db open failed: %s\n",
                db ? sqlite3_errmsg(db) : "null");
        if (db) sqlite3_close(db);
        return;
    }
    sqlite3_busy_timeout(db, 5000);

    sqlite3_exec(db,
        "CREATE TABLE IF NOT EXISTS clock_anchors ("
        "  phase TEXT, pid INTEGER,"
        "  cupti_ns INTEGER, host_ns INTEGER, bracket_ns INTEGER);",
        nullptr, nullptr, nullptr);

    if (g_anchor_init.valid)     insert_one(db, "init",     g_anchor_init);
    if (g_anchor_finalize.valid) insert_one(db, "finalize", g_anchor_finalize);

    sqlite3_close(db);
    fprintf(stderr, "[ANCHOR] written to %s\n", db_path);
    fflush(stderr);
}