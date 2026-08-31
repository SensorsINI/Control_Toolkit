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

/* Same Jacobian with caller-supplied cos(angle_next), sin(angle_next).
 * Pass the next-state trig already produced by the forward substep. */
void cartpole_model_substep_jacobian_with_trig(
    const RpgdConfig* cfg,
    const RpgdRuntime* runtime,
    const float* state6,
    float q,
    float dt,
    float cos_next,
    float sin_next,
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
