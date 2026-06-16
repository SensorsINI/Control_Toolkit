#include "cartpole_cost.h"

#include <math.h>

enum {
    ANGLE_IDX = 0,
    ANGLED_IDX = 1,
    ANGLE_COS_IDX = 2,
    ANGLE_SIN_IDX = 3,
    POSITION_IDX = 4,
    POSITIOND_IDX = 5
};

float cartpole_cost_stage(
    const RpgdConfig* c,
    const RpgdRuntime* rt,
    const float* s,
    float q
)
{
    const float target_eq = rt->target_equilibrium == 0.0f ? 1.0f : rt->target_equilibrium;
    const float L = rt->L > 0.0f ? rt->L : c->L;
    const float position = s[POSITION_IDX];
    const float angle = s[ANGLE_IDX];
    const float angleD = s[ANGLED_IDX];
    const float abs_pos = fabsf(position);
    const float T = c->track_half_length;

    const float dd = c->dd_quadratic_weight_up * powf((position - rt->target_position) / (2.0f * T), 2.0f);

    float db = 0.0f;
    const float boundary = c->permissible_track_fraction * T;
    if (abs_pos > boundary) {
        db = c->db_weight_up * powf((abs_pos - boundary) / ((1.0f - c->permissible_track_fraction) * T), 2.0f);
    }

    const float ep = c->ep_weight_up * powf(1.0f - target_eq * cosf(angle), 2.0f);
    const float up_only = (target_eq + 1.0f) * 0.5f;
    const float kinetic_ref = c->vel_penalty_reg * (3.0f * c->g / L) * (1.0f - cosf(angle));
    const float ekp = c->ekp_weight_up * up_only * fabsf(angleD * angleD - kinetic_ref);
    const float cc = c->cc_weight_up * c->R * q * q;
    return dd + db + ep + ekp + cc;
}

void cartpole_cost_stage_grad(
    const RpgdConfig* c,
    const RpgdRuntime* rt,
    const float* s,
    float q,
    float scale,
    float* gs,
    float* gq
)
{
    const float target_eq = rt->target_equilibrium == 0.0f ? 1.0f : rt->target_equilibrium;
    const float L = rt->L > 0.0f ? rt->L : c->L;
    const float T = c->track_half_length;
    const float position = s[POSITION_IDX];
    const float angle = s[ANGLE_IDX];
    const float angleD = s[ANGLED_IDX];

    gs[POSITION_IDX] += scale * c->dd_quadratic_weight_up * (position - rt->target_position) / (2.0f * T * T);

    const float abs_pos = fabsf(position);
    const float boundary = c->permissible_track_fraction * T;
    if (abs_pos > boundary) {
        const float denom = (1.0f - c->permissible_track_fraction) * T;
        const float sign = position >= 0.0f ? 1.0f : -1.0f;
        gs[POSITION_IDX] += scale * c->db_weight_up * 2.0f * (abs_pos - boundary) * sign / (denom * denom);
    }

    const float ep_inner = 1.0f - target_eq * cosf(angle);
    gs[ANGLE_IDX] += scale * c->ep_weight_up * 2.0f * ep_inner * target_eq * sinf(angle);

    const float up_only = (target_eq + 1.0f) * 0.5f;
    const float kinetic_ref_factor = c->vel_penalty_reg * (3.0f * c->g / L);
    const float kinetic_inner = angleD * angleD - kinetic_ref_factor * (1.0f - cosf(angle));
    const float kinetic_sign = kinetic_inner >= 0.0f ? 1.0f : -1.0f;
    gs[ANGLED_IDX] += scale * c->ekp_weight_up * up_only * kinetic_sign * 2.0f * angleD;
    gs[ANGLE_IDX] += scale * c->ekp_weight_up * up_only * kinetic_sign * (-kinetic_ref_factor * sinf(angle));

    *gq += scale * 2.0f * c->cc_weight_up * c->R * q;
}
