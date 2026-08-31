#ifndef CARTPOLE_COST_H
#define CARTPOLE_COST_H

#include "rpgd_cartpole.h"

float cartpole_cost_stage(
    const RpgdConfig* cfg,
    const RpgdRuntime* runtime,
    const float* state6,
    float q
);

void cartpole_cost_stage_grad(
    const RpgdConfig* cfg,
    const RpgdRuntime* runtime,
    const float* state6,
    float q,
    float scale,
    float* grad_state6,
    float* grad_q
);

void cartpole_cost_stage_grad_state(
    const RpgdConfig* cfg,
    const RpgdRuntime* runtime,
    const float* state6,
    float scale,
    float* grad_state6
);

float cartpole_cost_stage_grad_q(
    const RpgdConfig* cfg,
    float q,
    float scale
);

#endif
