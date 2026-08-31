#include "cartpole_model.h"
#include "rpgd_platform.h"

#if defined(RPGD_PLATFORM_BAREMETAL) && defined(__GNUC__)
#pragma GCC optimize("O3")
#endif

#include <math.h>
#include <string.h>

enum {
    ANGLE_IDX = 0,
    ANGLED_IDX = 1,
    ANGLE_COS_IDX = 2,
    ANGLE_SIN_IDX = 3,
    POSITION_IDX = 4,
    POSITIOND_IDX = 5
};

void cartpole_model_substep_dt(
    const RpgdConfig* c,
    const RpgdRuntime* rt,
    const float* s,
    float q,
    float dt,
    float* sn
)
{
    const float L = rt->L > 0.0f ? rt->L : c->L;
    const float m_pole = rt->m_pole > 0.0f ? rt->m_pole : c->m_pole;
    const float Lh = 0.5f * L;
    const float ca = s[ANGLE_COS_IDX];
    const float sa = s[ANGLE_SIN_IDX];
    const float angle = s[ANGLE_IDX];
    const float angleD = s[ANGLED_IDX];
    const float positionD = s[POSITIOND_IDX];
    const float u = c->u_max * q;

    const float F_fric = -c->M_fric * positionD;
    const float T_fric = -c->J_fric * angleD;
    const float kp1 = c->k + 1.0f;
    const float denom = kp1 * (c->m_cart + m_pole) - m_pole * ca * ca;
    const float positionDD = (
        m_pole * c->g * sa * ca
        + (T_fric * ca) / Lh
        + kp1 * (-m_pole * Lh * angleD * angleD * sa + F_fric + u)
    ) / denom;
    const float angleDD = (
        c->g * sa + positionDD * ca + T_fric / (m_pole * Lh)
    ) / (kp1 * Lh);

    const float w_next = angleD + angleDD * dt;
    const float v_next = positionD + positionDD * dt;
    const float a_next = angle + w_next * dt;
    const float x_next = s[POSITION_IDX] + v_next * dt;
    const float cos_next = cosf(a_next);
    const float sin_next = sinf(a_next);

    sn[ANGLE_IDX] = atan2f(sin_next, cos_next);
    sn[ANGLED_IDX] = w_next;
    sn[ANGLE_COS_IDX] = cos_next;
    sn[ANGLE_SIN_IDX] = sin_next;
    sn[POSITION_IDX] = x_next;
    sn[POSITIOND_IDX] = v_next;
}

void cartpole_model_substep_jacobian_with_trig(
    const RpgdConfig* c,
    const RpgdRuntime* rt,
    const float* s,
    float q,
    float dt,
    float cos_next,
    float sin_next,
    float Jx[6][6],
    float Ju[6]
)
{
    const float L = rt->L > 0.0f ? rt->L : c->L;
    const float m_pole = rt->m_pole > 0.0f ? rt->m_pole : c->m_pole;
    const float Lh = 0.5f * L;
    const float ca = s[ANGLE_COS_IDX];
    const float sa = s[ANGLE_SIN_IDX];
    const float angleD = s[ANGLED_IDX];
    const float positionD = s[POSITIOND_IDX];
    const float u = c->u_max * q;
    const float kp1 = c->k + 1.0f;

    const float F_fric = -c->M_fric * positionD;
    const float T_fric = -c->J_fric * angleD;
    const float denom = kp1 * (c->m_cart + m_pole) - m_pole * ca * ca;
    const float inv_denom = 1.0f / denom;
    const float num =
        m_pole * c->g * sa * ca
        + (T_fric * ca) / Lh
        + kp1 * (-m_pole * Lh * angleD * angleD * sa + F_fric + u);

    const float dden_dca = -2.0f * m_pole * ca;
    const float dnum_dca = m_pole * c->g * sa + T_fric / Lh;
    const float dnum_dsa = m_pole * c->g * ca + kp1 * (-m_pole * Lh * angleD * angleD);
    const float dnum_dw = (-c->J_fric * ca) / Lh + kp1 * (-2.0f * m_pole * Lh * angleD * sa);
    const float dnum_dv = -kp1 * c->M_fric;
    const float dnum_dq = kp1 * c->u_max;

    const float positionDD = num * inv_denom;
    const float dpos_dca = (dnum_dca * denom - num * dden_dca) * inv_denom * inv_denom;
    const float dpos_dsa = dnum_dsa * inv_denom;
    const float dpos_dw = dnum_dw * inv_denom;
    const float dpos_dv = dnum_dv * inv_denom;
    const float dpos_dq = dnum_dq * inv_denom;

    const float angle_den = kp1 * Lh;
    const float inv_angle_den = 1.0f / angle_den;
    const float dangle_dca = (dpos_dca * ca + positionDD) * inv_angle_den;
    const float dangle_dsa = (c->g + dpos_dsa * ca) * inv_angle_den;
    const float dangle_dw = (dpos_dw * ca - c->J_fric / (m_pole * Lh)) * inv_angle_den;
    const float dangle_dv = (dpos_dv * ca) * inv_angle_den;
    const float dangle_dq = (dpos_dq * ca) * inv_angle_den;

    float dw[6] = {0};
    float dv[6] = {0};
    dw[ANGLED_IDX] = 1.0f + dt * dangle_dw;
    dw[ANGLE_COS_IDX] = dt * dangle_dca;
    dw[ANGLE_SIN_IDX] = dt * dangle_dsa;
    dw[POSITIOND_IDX] = dt * dangle_dv;

    dv[ANGLED_IDX] = dt * dpos_dw;
    dv[ANGLE_COS_IDX] = dt * dpos_dca;
    dv[ANGLE_SIN_IDX] = dt * dpos_dsa;
    dv[POSITIOND_IDX] = 1.0f + dt * dpos_dv;

    const float dw_q = dt * dangle_dq;
    const float dv_q = dt * dpos_dq;
    const float da_q = dt * dw_q;
    const float dx_q = dt * dv_q;

    float da[6] = {0};
    float dx[6] = {0};
    for (int j = 0; j < 6; ++j) {
        da[j] = dt * dw[j];
        dx[j] = dt * dv[j];
    }
    da[ANGLE_IDX] += 1.0f;
    dx[POSITION_IDX] += 1.0f;

    for (int j = 0; j < 6; ++j) {
        Jx[ANGLE_IDX][j] = da[j];
        Jx[ANGLED_IDX][j] = dw[j];
        Jx[ANGLE_COS_IDX][j] = -sin_next * da[j];
        Jx[ANGLE_SIN_IDX][j] = cos_next * da[j];
        Jx[POSITION_IDX][j] = dx[j];
        Jx[POSITIOND_IDX][j] = dv[j];
    }
    Ju[ANGLE_IDX] = da_q;
    Ju[ANGLED_IDX] = dw_q;
    Ju[ANGLE_COS_IDX] = -sin_next * da_q;
    Ju[ANGLE_SIN_IDX] = cos_next * da_q;
    Ju[POSITION_IDX] = dx_q;
    Ju[POSITIOND_IDX] = dv_q;
}

void cartpole_model_substep_jacobian(
    const RpgdConfig* c,
    const RpgdRuntime* rt,
    const float* s,
    float q,
    float dt,
    float Jx[6][6],
    float Ju[6]
)
{
    const float L = rt->L > 0.0f ? rt->L : c->L;
    const float m_pole = rt->m_pole > 0.0f ? rt->m_pole : c->m_pole;
    const float Lh = 0.5f * L;
    const float ca = s[ANGLE_COS_IDX];
    const float sa = s[ANGLE_SIN_IDX];
    const float angleD = s[ANGLED_IDX];
    const float positionD = s[POSITIOND_IDX];
    const float u = c->u_max * q;
    const float kp1 = c->k + 1.0f;

    const float F_fric = -c->M_fric * positionD;
    const float T_fric = -c->J_fric * angleD;
    const float denom = kp1 * (c->m_cart + m_pole) - m_pole * ca * ca;
    const float inv_denom = 1.0f / denom;
    const float num =
        m_pole * c->g * sa * ca
        + (T_fric * ca) / Lh
        + kp1 * (-m_pole * Lh * angleD * angleD * sa + F_fric + u);
    const float positionDD = num * inv_denom;
    const float inv_angle_den = 1.0f / (kp1 * Lh);
    const float w_next = s[ANGLED_IDX] + dt * (
        c->g * sa + positionDD * ca + T_fric / (m_pole * Lh)
    ) * inv_angle_den;
    const float angle_next = s[ANGLE_IDX] + dt * w_next;
    cartpole_model_substep_jacobian_with_trig(
        c, rt, s, q, dt, cosf(angle_next), sinf(angle_next), Jx, Ju);
}

void cartpole_model_rollout_final_state(
    const RpgdConfig* cfg,
    const RpgdRuntime* runtime,
    const float* state6,
    const float* q,
    float* final_state6
)
{
    float s[6];
    float sn[6];
    const float dt = cfg->mpc_timestep / (float)cfg->intermediate_steps;
    memcpy(s, state6, 6 * sizeof(float));
    for (int h = 0; h < cfg->mpc_horizon; ++h) {
        for (int k = 0; k < cfg->intermediate_steps; ++k) {
            cartpole_model_substep_dt(cfg, runtime, s, q[h], dt, sn);
            memcpy(s, sn, 6 * sizeof(float));
        }
    }
    memcpy(final_state6, s, 6 * sizeof(float));
}
