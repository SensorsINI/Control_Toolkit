#ifndef RPGD_WORKER_H
#define RPGD_WORKER_H

/*
 * Internal deterministic phase API for split / dual-core execution.
 * Not an AMP interface: mailbox and core affinity live in firmware.
 *
 * rpgd_step() is prepare -> optimize_range(0, N) -> finalize on the
 * single-thread path, and must stay bit-identical to the previous
 * monolithic implementation.
 */

#include "rpgd_cartpole.h"
#include "rpgd_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RpgdWorkerScratch {
    float states[RPGD_MAX_STATE_BUF];
    float grad[RPGD_MAX_HORIZON];
} RPGD_ALIGN64 RpgdWorkerScratch;

typedef struct RpgdStepPlan {
    float state6[6];
    RpgdRuntime runtime;
    int active_iterations;
    int prepared;
    int range_error;
} RpgdStepPlan;

int rpgd_step_prepare(RpgdSolver* solver, const float state6[6],
                      const RpgdRuntime* runtime, RpgdStepPlan* plan);
int rpgd_step_optimize_range(RpgdSolver* solver, const RpgdStepPlan* plan,
                             int first, int last,
                             RpgdWorkerScratch* scratch);
float rpgd_step_finalize(RpgdSolver* solver, RpgdStepPlan* plan);
void rpgd_step_abort(RpgdSolver* solver, int status);

void rpgd_cache_visit_solver(RpgdSolver* solver, void (*fn)(const void*, size_t));
void rpgd_cache_visit_rollout_slice(RpgdSolver* solver, int first, int last,
                                    void (*fn)(const void*, size_t));

#ifdef __cplusplus
}
#endif

#endif
