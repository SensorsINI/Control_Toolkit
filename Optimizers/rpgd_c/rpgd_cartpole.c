#include "rpgd_cartpole.h"
#include "cartpole_cost.h"
#include "cartpole_model.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#else
#include <pthread.h>
#include <unistd.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define STATE_DIM 6

enum {
    ANGLE_IDX = 0,
    ANGLED_IDX = 1,
    ANGLE_COS_IDX = 2,
    ANGLE_SIN_IDX = 3,
    POSITION_IDX = 4,
    POSITIOND_IDX = 5
};

typedef struct {
    uint64_t state;
    int has_spare;
    float spare;
} RngState;

struct RpgdSolver {
    RpgdConfig cfg;
    int opt_keep_k;
    int inducing_points;
    int thread_count;
    int count;
    int adam_step;
    float dt_sub;

    float* q;
    float* adam_m;
    float* adam_v;
    float* trajectory_ages;
    float* costs;
    int* indices;
    RngState rng;
};

#ifndef _OPENMP
typedef struct {
    RpgdSolver* solver;
    const RpgdRuntime* runtime;
    const float* state6;
    int start;
    int end;
} RpgdThreadArgs;
#endif

static float clampf_local(float x, float lo, float hi)
{
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static uint64_t splitmix64(uint64_t* x)
{
    uint64_t z = (*x += UINT64_C(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

static void rng_seed(RngState* rng, unsigned int seed)
{
    uint64_t s = seed ? seed : 1u;
    rng->state = splitmix64(&s);
    rng->has_spare = 0;
    rng->spare = 0.0f;
}

static uint32_t rng_u32(RngState* rng)
{
    uint64_t x = rng->state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng->state = x;
    return (uint32_t)((x * UINT64_C(2685821657736338717)) >> 32);
}

static float rng_uniform01(RngState* rng)
{
    return ((float)rng_u32(rng) + 0.5f) * (1.0f / 4294967296.0f);
}

static float rng_normal(RngState* rng, float mean, float stdev)
{
    if (rng->has_spare) {
        rng->has_spare = 0;
        return mean + stdev * rng->spare;
    }
    float u1 = rng_uniform01(rng);
    float u2 = rng_uniform01(rng);
    float r = sqrtf(-2.0f * logf(fmaxf(u1, 1.0e-12f)));
    float theta = 2.0f * (float)M_PI * u2;
    rng->spare = r * sinf(theta);
    rng->has_spare = 1;
    return mean + stdev * (r * cosf(theta));
}

static int calc_inducing_points(int horizon, int period)
{
    if (period <= 1) return horizon;
    return (int)ceilf(((float)horizon - 1.0f) / (float)period) + 1;
}

static void sample_action_sequence(RpgdSolver* solver, float* out_q)
{
    const RpgdConfig* c = &solver->cfg;
    const int h = c->mpc_horizon;
    const int p = c->period_interpolation_inducing_points;
    const int n = solver->inducing_points;
    float* points = (float*)malloc((size_t)n * sizeof(float));
    if (!points) return;

    for (int i = 0; i < n; ++i) {
        float v;
        if (c->sampling_distribution == 1) {
            const float lo = c->sample_whole_control_space ? c->action_low : c->uniform_dist_min;
            const float hi = c->sample_whole_control_space ? c->action_high : c->uniform_dist_max;
            v = lo + (hi - lo) * rng_uniform01(&solver->rng);
        } else {
            v = rng_normal(&solver->rng, c->sample_mean, c->sample_stdev);
        }
        points[i] = clampf_local(v, c->action_low, c->action_high);
    }

    if (p <= 1) {
        for (int i = 0; i < h; ++i) out_q[i] = points[i];
    } else {
        for (int t = 0; t < h; ++t) {
            int left = t / p;
            int right = left + 1;
            if (right >= n) right = n - 1;
            const float frac = (float)(t - left * p) / (float)p;
            out_q[t] = (1.0f - frac) * points[left] + frac * points[right];
        }
    }
    free(points);
}

static void sort_indices_by_cost(const float* costs, int* idx, int n)
{
    for (int i = 0; i < n; ++i) idx[i] = i;
    for (int i = 1; i < n; ++i) {
        int key = idx[i];
        float v = costs[key];
        int j = i - 1;
        while (j >= 0 && costs[idx[j]] > v) {
            idx[j + 1] = idx[j];
            --j;
        }
        idx[j + 1] = key;
    }
}

static void cartpole_substep(const RpgdConfig* c, const RpgdRuntime* rt, const float* s, float q, float* sn)
{
    const float L = rt->L > 0.0f ? rt->L : c->L;
    const float m_pole = rt->m_pole > 0.0f ? rt->m_pole : c->m_pole;
    const float Lh = 0.5f * L;
    const float ca = s[ANGLE_COS_IDX];
    const float sa = s[ANGLE_SIN_IDX];
    const float angle = s[ANGLE_IDX];
    const float angleD = s[ANGLED_IDX];
    const float position = s[POSITION_IDX];
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

    const float w_next = angleD + angleDD * c->mpc_timestep / (float)c->intermediate_steps;
    const float v_next = positionD + positionDD * c->mpc_timestep / (float)c->intermediate_steps;
    const float a_next = angle + w_next * c->mpc_timestep / (float)c->intermediate_steps;
    const float x_next = position + v_next * c->mpc_timestep / (float)c->intermediate_steps;
    const float cos_next = cosf(a_next);
    const float sin_next = sinf(a_next);

    sn[ANGLE_IDX] = atan2f(sin_next, cos_next);
    sn[ANGLED_IDX] = w_next;
    sn[ANGLE_COS_IDX] = cos_next;
    sn[ANGLE_SIN_IDX] = sin_next;
    sn[POSITION_IDX] = x_next;
    sn[POSITIOND_IDX] = v_next;
}

static void cartpole_substep_dt(const RpgdConfig* c, const RpgdRuntime* rt, const float* s, float q, float dt, float* sn)
{
    RpgdConfig tmp = *c;
    tmp.mpc_timestep = dt;
    tmp.intermediate_steps = 1;
    cartpole_substep(&tmp, rt, s, q, sn);
}

static float stage_cost(const RpgdConfig* c, const RpgdRuntime* rt, const float* s, float q)
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

static void stage_cost_grad(const RpgdConfig* c, const RpgdRuntime* rt, const float* s, float q, float scale, float* gs, float* gq)
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

static float rollout_cost(const RpgdConfig* c, const RpgdRuntime* rt, const float* state6, const float* q)
{
    float s[STATE_DIM];
    float sn[STATE_DIM];
    memcpy(s, state6, STATE_DIM * sizeof(float));

    float total = 0.0f;
    for (int h = 0; h < c->mpc_horizon; ++h) {
        total += cartpole_cost_stage(c, rt, s, q[h]);
        for (int k = 0; k < c->intermediate_steps; ++k) {
            cartpole_model_substep_dt(c, rt, s, q[h], c->mpc_timestep / (float)c->intermediate_steps, sn);
            memcpy(s, sn, STATE_DIM * sizeof(float));
        }
    }
    return total / (float)(c->mpc_horizon + 1);
}

static void rollout_gradient_fd_eps(const RpgdConfig* c, const RpgdRuntime* rt, const float* state6, const float* q, float eps, float* grad)
{
    float* qp = (float*)malloc((size_t)c->mpc_horizon * sizeof(float));
    if (!qp) return;
    memcpy(qp, q, (size_t)c->mpc_horizon * sizeof(float));
    for (int h = 0; h < c->mpc_horizon; ++h) {
        const float old = qp[h];
        qp[h] = clampf_local(old + eps, c->action_low, c->action_high);
        const float cp = rollout_cost(c, rt, state6, qp);
        qp[h] = clampf_local(old - eps, c->action_low, c->action_high);
        const float cm = rollout_cost(c, rt, state6, qp);
        qp[h] = old;
        grad[h] = (cp - cm) / (2.0f * eps);
    }
    free(qp);
}

static void rollout_gradient_fd(const RpgdConfig* c, const RpgdRuntime* rt, const float* state6, const float* q, float* grad)
{
    rollout_gradient_fd_eps(c, rt, state6, q, 1.0e-2f, grad);
}

static void local_jacobian_analytic(const RpgdConfig* c, const RpgdRuntime* rt, const float* s, float q, float dt, float Jx[STATE_DIM][STATE_DIM], float Ju[STATE_DIM])
{
    memset(Jx, 0, STATE_DIM * STATE_DIM * sizeof(float));
    memset(Ju, 0, STATE_DIM * sizeof(float));

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

    float dw[STATE_DIM] = {0};
    float dv[STATE_DIM] = {0};
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

    float da[STATE_DIM] = {0};
    float dx[STATE_DIM] = {0};
    for (int j = 0; j < STATE_DIM; ++j) {
        da[j] = dt * dw[j];
        dx[j] = dt * dv[j];
    }
    da[ANGLE_IDX] += 1.0f;
    dx[POSITION_IDX] += 1.0f;

    const float w_next = s[ANGLED_IDX] + dt * (
        c->g * sa + positionDD * ca + T_fric / (m_pole * Lh)
    ) * inv_angle_den;
    const float angle_next = s[ANGLE_IDX] + dt * w_next;
    const float cos_next = cosf(angle_next);
    const float sin_next = sinf(angle_next);

    for (int j = 0; j < STATE_DIM; ++j) {
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

static void rollout_gradient_adjoint(const RpgdConfig* c, const RpgdRuntime* rt, const float* state6, const float* q, float* grad)
{
    const int total_steps = c->mpc_horizon * c->intermediate_steps;
    float* states = (float*)calloc((size_t)(total_steps + 1) * STATE_DIM, sizeof(float));
    float* adj = (float*)calloc((size_t)(total_steps + 1) * STATE_DIM, sizeof(float));
    if (!states || !adj) {
        free(states);
        free(adj);
        rollout_gradient_fd(c, rt, state6, q, grad);
        return;
    }
    memset(grad, 0, (size_t)c->mpc_horizon * sizeof(float));
    memcpy(states, state6, STATE_DIM * sizeof(float));
    for (int n = 0; n < total_steps; ++n) {
        int h = n / c->intermediate_steps;
        cartpole_model_substep_dt(c, rt, &states[(size_t)n * STATE_DIM], q[h],
                                  c->mpc_timestep / (float)c->intermediate_steps,
                                  &states[(size_t)(n + 1) * STATE_DIM]);
    }

    const float scale = 1.0f / (float)(c->mpc_horizon + 1);
    for (int h = 0; h < c->mpc_horizon; ++h) {
        const int n = h * c->intermediate_steps;
        cartpole_cost_stage_grad(c, rt, &states[(size_t)n * STATE_DIM], q[h], scale,
                                 &adj[(size_t)n * STATE_DIM], &grad[h]);
    }

    for (int n = total_steps - 1; n >= 0; --n) {
        const int h = n / c->intermediate_steps;
        float Jx[STATE_DIM][STATE_DIM];
        float Ju[STATE_DIM];
        cartpole_model_substep_jacobian(c, rt, &states[(size_t)n * STATE_DIM], q[h],
                                        c->mpc_timestep / (float)c->intermediate_steps, Jx, Ju);
        float* a_next = &adj[(size_t)(n + 1) * STATE_DIM];
        float* a_cur = &adj[(size_t)n * STATE_DIM];
        for (int j = 0; j < STATE_DIM; ++j) {
            float acc = 0.0f;
            for (int i = 0; i < STATE_DIM; ++i) acc += a_next[i] * Jx[i][j];
            a_cur[j] += acc;
        }
        float gu = 0.0f;
        for (int i = 0; i < STATE_DIM; ++i) gu += a_next[i] * Ju[i];
        grad[h] += gu;
    }

    free(states);
    free(adj);
}

static void clip_gradient_norm(const RpgdConfig* c, float* grad)
{
    float sum = 0.0f;
    for (int h = 0; h < c->mpc_horizon; ++h) sum += grad[h] * grad[h];
    const float norm = sqrtf(sum);
    if (norm > c->gradmax_clip && norm > 0.0f) {
        const float scale = c->gradmax_clip / norm;
        for (int h = 0; h < c->mpc_horizon; ++h) grad[h] *= scale;
    }
}

static void adam_update_rollout(RpgdSolver* solver, int rollout, const float* grad, int adam_step)
{
    RpgdConfig* c = &solver->cfg;
    const int H = c->mpc_horizon;
    float* q = &solver->q[(size_t)rollout * H];
    float* m = &solver->adam_m[(size_t)rollout * H];
    float* v = &solver->adam_v[(size_t)rollout * H];
    const float bc1 = 1.0f - powf(c->adam_beta_1, (float)adam_step);
    const float bc2 = 1.0f - powf(c->adam_beta_2, (float)adam_step);
    for (int h = 0; h < H; ++h) {
        m[h] = c->adam_beta_1 * m[h] + (1.0f - c->adam_beta_1) * grad[h];
        v[h] = c->adam_beta_2 * v[h] + (1.0f - c->adam_beta_2) * grad[h] * grad[h];
        const float m_hat = m[h] / bc1;
        const float v_hat = v[h] / bc2;
        q[h] = clampf_local(q[h] - c->learning_rate * m_hat / (sqrtf(v_hat) + c->adam_epsilon),
                            c->action_low, c->action_high);
    }
}

static void optimize_rollout(RpgdSolver* solver, const RpgdRuntime* rt, const float* state6, int rollout)
{
    const int H = solver->cfg.mpc_horizon;
    float* grad = (float*)malloc((size_t)H * sizeof(float));
    if (!grad) return;
    float* q = &solver->q[(size_t)rollout * H];
    for (int it = 0; it < solver->cfg.outer_its; ++it) {
        rollout_gradient_adjoint(&solver->cfg, rt, state6, q, grad);
        clip_gradient_norm(&solver->cfg, grad);
        adam_update_rollout(solver, rollout, grad, solver->adam_step + it + 1);
    }
    solver->costs[rollout] = rollout_cost(&solver->cfg, rt, state6, q);
    free(grad);
}

#ifndef _OPENMP
static void* optimize_rollout_range(void* arg)
{
    RpgdThreadArgs* a = (RpgdThreadArgs*)arg;
    for (int i = a->start; i < a->end; ++i) {
        optimize_rollout(a->solver, a->runtime, a->state6, i);
    }
    return NULL;
}
#endif

static void warmstart_after_action(RpgdSolver* solver)
{
    RpgdConfig* c = &solver->cfg;
    const int N = c->num_rollouts;
    const int H = c->mpc_horizon;
    const int K = solver->opt_keep_k;
    const int do_resample = (solver->count % c->resamp_per) == 0;
    float* qn = (float*)malloc((size_t)N * H * sizeof(float));
    float* mn = (float*)calloc((size_t)N * H, sizeof(float));
    float* vn = (float*)calloc((size_t)N * H, sizeof(float));
    float* ages = (float*)calloc((size_t)N, sizeof(float));
    if (!qn || !mn || !vn || !ages) {
        free(qn); free(mn); free(vn); free(ages);
        return;
    }

    int dst = 0;
    if (do_resample) {
        for (; dst < N - K; ++dst) {
            sample_action_sequence(solver, &qn[(size_t)dst * H]);
            ages[dst] = 0.0f;
        }
        for (int k = 0; k < K; ++k, ++dst) {
            const int src = solver->indices[k];
            float* dq = &qn[(size_t)dst * H];
            float* dm = &mn[(size_t)dst * H];
            float* dv = &vn[(size_t)dst * H];
            const float* sq = &solver->q[(size_t)src * H];
            const float* sm = &solver->adam_m[(size_t)src * H];
            const float* sv = &solver->adam_v[(size_t)src * H];
            for (int h = 0; h < H; ++h) {
                int sh = h + c->shift_previous;
                if (sh >= H) sh = H - 1;
                dq[h] = sq[sh];
                dm[h] = (h + c->shift_previous < H) ? sm[h + c->shift_previous] : 0.0f;
                dv[h] = (h + c->shift_previous < H) ? sv[h + c->shift_previous] : 0.0f;
            }
            ages[dst] = solver->trajectory_ages[src];
        }
    } else {
        for (int src = 0; src < N; ++src) {
            float* dq = &qn[(size_t)src * H];
            float* dm = &mn[(size_t)src * H];
            float* dv = &vn[(size_t)src * H];
            const float* sq = &solver->q[(size_t)src * H];
            const float* sm = &solver->adam_m[(size_t)src * H];
            const float* sv = &solver->adam_v[(size_t)src * H];
            for (int h = 0; h < H; ++h) {
                int sh = h + c->shift_previous;
                if (sh >= H) sh = H - 1;
                dq[h] = sq[sh];
                dm[h] = (h + c->shift_previous < H) ? sm[h + c->shift_previous] : 0.0f;
                dv[h] = (h + c->shift_previous < H) ? sv[h + c->shift_previous] : 0.0f;
            }
            ages[src] = solver->trajectory_ages[src];
        }
    }
    for (int i = 0; i < N; ++i) ages[i] += 1.0f;
    memcpy(solver->q, qn, (size_t)N * H * sizeof(float));
    memcpy(solver->adam_m, mn, (size_t)N * H * sizeof(float));
    memcpy(solver->adam_v, vn, (size_t)N * H * sizeof(float));
    memcpy(solver->trajectory_ages, ages, (size_t)N * sizeof(float));
    free(qn); free(mn); free(vn); free(ages);
}

static int choose_thread_count(const RpgdConfig* cfg)
{
    int available = 1;
#ifdef _OPENMP
    available = omp_get_num_procs();
#else
    long nproc = sysconf(_SC_NPROCESSORS_ONLN);
    if (nproc > 0) available = (int)nproc;
#endif
    int threads = cfg->num_threads;
    if (threads <= 0) {
        threads = available - cfg->reserve_threads;
        if (threads < 1) threads = 1;
    }
    if (threads > cfg->num_rollouts) threads = cfg->num_rollouts;
    if (threads < 1) threads = 1;
    return threads;
}

RpgdSolver* rpgd_create(const RpgdConfig* cfg)
{
    if (!cfg || cfg->mpc_horizon <= 0 || cfg->num_rollouts <= 0) return NULL;
    RpgdSolver* solver = (RpgdSolver*)calloc(1, sizeof(RpgdSolver));
    if (!solver) return NULL;
    solver->cfg = *cfg;
    if (solver->cfg.period_interpolation_inducing_points <= 0) solver->cfg.period_interpolation_inducing_points = 1;
    if (solver->cfg.intermediate_steps <= 0) solver->cfg.intermediate_steps = 10;
    if (solver->cfg.resamp_per <= 0) solver->cfg.resamp_per = 1;
    if (solver->cfg.shift_previous <= 0) solver->cfg.shift_previous = 1;
    if (solver->cfg.reserve_threads < 0) solver->cfg.reserve_threads = 0;
    solver->opt_keep_k = (int)(cfg->num_rollouts * cfg->opt_keep_k_ratio);
    if (solver->opt_keep_k < 1) solver->opt_keep_k = 1;
    if (solver->opt_keep_k > cfg->num_rollouts) solver->opt_keep_k = cfg->num_rollouts;
    solver->inducing_points = calc_inducing_points(cfg->mpc_horizon, solver->cfg.period_interpolation_inducing_points);
    solver->thread_count = choose_thread_count(&solver->cfg);
    const size_t q_size = (size_t)cfg->num_rollouts * cfg->mpc_horizon;
    solver->q = (float*)calloc(q_size, sizeof(float));
    solver->adam_m = (float*)calloc(q_size, sizeof(float));
    solver->adam_v = (float*)calloc(q_size, sizeof(float));
    solver->trajectory_ages = (float*)calloc((size_t)cfg->num_rollouts, sizeof(float));
    solver->costs = (float*)calloc((size_t)cfg->num_rollouts, sizeof(float));
    solver->indices = (int*)calloc((size_t)cfg->num_rollouts, sizeof(int));
    if (!solver->q || !solver->adam_m || !solver->adam_v || !solver->trajectory_ages || !solver->costs || !solver->indices) {
        rpgd_destroy(solver);
        return NULL;
    }
    rpgd_reset(solver, cfg->seed);
    return solver;
}

void rpgd_destroy(RpgdSolver* solver)
{
    if (!solver) return;
    free(solver->q);
    free(solver->adam_m);
    free(solver->adam_v);
    free(solver->trajectory_ages);
    free(solver->costs);
    free(solver->indices);
    free(solver);
}

void rpgd_reset(RpgdSolver* solver, unsigned int seed)
{
    if (!solver) return;
    rng_seed(&solver->rng, seed ? seed : solver->cfg.seed);
    solver->count = 0;
    solver->adam_step = 0;
    const int N = solver->cfg.num_rollouts;
    const int H = solver->cfg.mpc_horizon;
    for (int i = 0; i < N; ++i) sample_action_sequence(solver, &solver->q[(size_t)i * H]);
    memset(solver->adam_m, 0, (size_t)N * H * sizeof(float));
    memset(solver->adam_v, 0, (size_t)N * H * sizeof(float));
    memset(solver->trajectory_ages, 0, (size_t)N * sizeof(float));
}

float rpgd_step(RpgdSolver* solver, const float* state6, const RpgdRuntime* runtime)
{
    if (!solver || !state6 || !runtime) return 0.0f;
    const int N = solver->cfg.num_rollouts;
#ifdef _OPENMP
    omp_set_num_threads(solver->thread_count);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        optimize_rollout(solver, runtime, state6, i);
    }
#else
    if (solver->thread_count <= 1) {
        for (int i = 0; i < N; ++i) {
            optimize_rollout(solver, runtime, state6, i);
        }
    } else {
        pthread_t* threads = (pthread_t*)calloc((size_t)solver->thread_count, sizeof(pthread_t));
        RpgdThreadArgs* args = (RpgdThreadArgs*)calloc((size_t)solver->thread_count, sizeof(RpgdThreadArgs));
        if (!threads || !args) {
            free(threads);
            free(args);
            for (int i = 0; i < N; ++i) {
                optimize_rollout(solver, runtime, state6, i);
            }
        } else {
            for (int t = 0; t < solver->thread_count; ++t) {
                const int start = (N * t) / solver->thread_count;
                const int end = (N * (t + 1)) / solver->thread_count;
                args[t].solver = solver;
                args[t].runtime = runtime;
                args[t].state6 = state6;
                args[t].start = start;
                args[t].end = end;
                pthread_create(&threads[t], NULL, optimize_rollout_range, &args[t]);
            }
            for (int t = 0; t < solver->thread_count; ++t) {
                pthread_join(threads[t], NULL);
            }
            free(threads);
            free(args);
        }
    }
#endif
    sort_indices_by_cost(solver->costs, solver->indices, N);
    solver->adam_step += solver->cfg.outer_its;
    const float u = solver->q[(size_t)solver->indices[0] * solver->cfg.mpc_horizon];
    warmstart_after_action(solver);
    solver->count += 1;
    return clampf_local(u, solver->cfg.action_low, solver->cfg.action_high);
}

int rpgd_gradient_check(
    const RpgdConfig* cfg,
    const RpgdRuntime* runtime,
    const float* state6,
    const float* q,
    float eps,
    float* max_abs_error,
    float* max_rel_error
)
{
    if (!cfg || !runtime || !state6 || !q || !max_abs_error || !max_rel_error) return -1;
    float* ga = (float*)calloc((size_t)cfg->mpc_horizon, sizeof(float));
    float* gf = (float*)calloc((size_t)cfg->mpc_horizon, sizeof(float));
    if (!ga || !gf) {
        free(ga); free(gf);
        return -2;
    }
    rollout_gradient_adjoint(cfg, runtime, state6, q, ga);
    rollout_gradient_fd_eps(cfg, runtime, state6, q, eps > 0.0f ? eps : 1.0e-2f, gf);
    float ma = 0.0f;
    float mr = 0.0f;
    for (int h = 0; h < cfg->mpc_horizon; ++h) {
        float abs_err = fabsf(ga[h] - gf[h]);
        float rel_err = abs_err / fmaxf(1.0f, fabsf(gf[h]));
        if (abs_err > ma) ma = abs_err;
        if (rel_err > mr) mr = rel_err;
    }
    *max_abs_error = ma;
    *max_rel_error = mr;
    free(ga); free(gf);
    return 0;
}

float rpgd_debug_rollout_cost(
    const RpgdConfig* cfg,
    const RpgdRuntime* runtime,
    const float* state6,
    const float* q
)
{
    if (!cfg || !runtime || !state6 || !q) return NAN;
    return rollout_cost(cfg, runtime, state6, q);
}

void rpgd_debug_rollout_final_state(
    const RpgdConfig* cfg,
    const RpgdRuntime* runtime,
    const float* state6,
    const float* q,
    float* final_state6
)
{
    if (!cfg || !runtime || !state6 || !q || !final_state6) return;
    cartpole_model_rollout_final_state(cfg, runtime, state6, q, final_state6);
}

void rpgd_debug_gradient_adjoint(
    const RpgdConfig* cfg,
    const RpgdRuntime* runtime,
    const float* state6,
    const float* q,
    float* grad
)
{
    if (!cfg || !runtime || !state6 || !q || !grad) return;
    rollout_gradient_adjoint(cfg, runtime, state6, q, grad);
}

void rpgd_debug_gradient_fd(
    const RpgdConfig* cfg,
    const RpgdRuntime* runtime,
    const float* state6,
    const float* q,
    float* grad
)
{
    if (!cfg || !runtime || !state6 || !q || !grad) return;
    rollout_gradient_fd(cfg, runtime, state6, q, grad);
}

void rpgd_debug_set_q(RpgdSolver* solver, const float* q)
{
    if (!solver || !q) return;
    memcpy(solver->q, q, (size_t)solver->cfg.num_rollouts * solver->cfg.mpc_horizon * sizeof(float));
}

void rpgd_debug_get_q(const RpgdSolver* solver, float* q)
{
    if (!solver || !q) return;
    memcpy(q, solver->q, (size_t)solver->cfg.num_rollouts * solver->cfg.mpc_horizon * sizeof(float));
}

void rpgd_debug_set_adam(RpgdSolver* solver, const float* m, const float* v, int step)
{
    if (!solver || !m || !v) return;
    const size_t n = (size_t)solver->cfg.num_rollouts * solver->cfg.mpc_horizon;
    memcpy(solver->adam_m, m, n * sizeof(float));
    memcpy(solver->adam_v, v, n * sizeof(float));
    solver->adam_step = step;
}

void rpgd_debug_get_adam(const RpgdSolver* solver, float* m, float* v, int* step)
{
    if (!solver) return;
    const size_t n = (size_t)solver->cfg.num_rollouts * solver->cfg.mpc_horizon;
    if (m) memcpy(m, solver->adam_m, n * sizeof(float));
    if (v) memcpy(v, solver->adam_v, n * sizeof(float));
    if (step) *step = solver->adam_step;
}

void rpgd_debug_get_costs(const RpgdSolver* solver, float* costs)
{
    if (!solver || !costs) return;
    memcpy(costs, solver->costs, (size_t)solver->cfg.num_rollouts * sizeof(float));
}

void rpgd_debug_get_indices(const RpgdSolver* solver, int* indices)
{
    if (!solver || !indices) return;
    memcpy(indices, solver->indices, (size_t)solver->cfg.num_rollouts * sizeof(int));
}

int rpgd_get_num_threads(const RpgdSolver* solver)
{
    return solver ? solver->thread_count : 0;
}

int rpgd_get_num_rollouts(const RpgdSolver* solver)
{
    return solver ? solver->cfg.num_rollouts : 0;
}

int rpgd_get_horizon(const RpgdSolver* solver)
{
    return solver ? solver->cfg.mpc_horizon : 0;
}
