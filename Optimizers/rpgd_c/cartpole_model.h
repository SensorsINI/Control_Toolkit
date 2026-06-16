#ifndef CARTPOLE_MODEL_H
#define CARTPOLE_MODEL_H

#include "rpgd_cartpole.h"

void cartpole_model_substep_dt(
    const RpgdConfig* cfg,
    const RpgdRuntime* runtime,
    const float* state6,
    float q,
    float dt,
    float* next_state6
);

void cartpole_model_substep_jacobian(
    const RpgdConfig* cfg,
    const RpgdRuntime* runtime,
    const float* state6,
    float q,
    float dt,
    float Jx[6][6],
    float Ju[6]
);

void cartpole_model_rollout_final_state(
    const RpgdConfig* cfg,
    const RpgdRuntime* runtime,
    const float* state6,
    const float* q,
    float* final_state6
);

#endif
