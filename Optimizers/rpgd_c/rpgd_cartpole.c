#include "rpgd_cartpole.h"
#include "rpgd_worker.h"
#include "cartpole_cost.h"
#include "cartpole_model.h"
#include "rpgd_platform.h"

#if defined(RPGD_PLATFORM_BAREMETAL) && defined(__GNUC__)
#pragma GCC optimize("O3")
#endif

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#ifndef RPGD_PLATFORM_BAREMETAL
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#else
#include <pthread.h>
#include <unistd.h>
#endif
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef NAN
#define NAN (0.0f / 0.0f)
#endif

#define STATE_DIM 6
#define RPGD_ABI_VERSION 1u

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
    int resample_phase;
    int active_iterations;
    int first_step;
    uint64_t adam_step;
    int last_status;
    int busy;
    int owns_storage;
    size_t workspace_bytes;
    float one_minus_beta1;
    float one_minus_beta2;

    float* q;
    float* adam_m;
    float* adam_v;
    float* trajectory_ages;
    float* costs;
    int* indices;
    float* inducing;
    float* warm_q;
    float* warm_m;
    float* warm_v;
    float* warm_ages;
    float* bias_correction_1;
    float* bias_correction_2;
    RngState rng;
};

#if !defined(RPGD_PLATFORM_BAREMETAL) && !defined(_OPENMP)
typedef struct {
    RpgdSolver* solver;
    const RpgdRuntime* runtime;
    const float* state6;
    int start;
    int end;
} RpgdThreadArgs;
#endif

#ifdef RPGD_PLATFORM_BAREMETAL
#ifndef RPGD_WORKER_ONLY
static RpgdSolver g_solver RPGD_SHARED;
static float g_q[RPGD_MAX_Q_BUF] RPGD_SHARED;
static float g_adam_m[RPGD_MAX_Q_BUF] RPGD_SHARED;
static float g_adam_v[RPGD_MAX_Q_BUF] RPGD_SHARED;
static float g_trajectory_ages[RPGD_MAX_NUM_ROLLOUTS] RPGD_SHARED;
static float g_costs[RPGD_MAX_NUM_ROLLOUTS] RPGD_SHARED;
static int g_indices[RPGD_MAX_NUM_ROLLOUTS] RPGD_SHARED;
static float g_inducing[RPGD_MAX_HORIZON] RPGD_SHARED;
static float g_warm_q[RPGD_MAX_Q_BUF] RPGD_SHARED;
static float g_warm_m[RPGD_MAX_Q_BUF] RPGD_SHARED;
static float g_warm_v[RPGD_MAX_Q_BUF] RPGD_SHARED;
static float g_warm_ages[RPGD_MAX_NUM_ROLLOUTS] RPGD_SHARED;
static float g_bias_correction_1[RPGD_MAX_OUTER_ITS] RPGD_SHARED;
static float g_bias_correction_2[RPGD_MAX_OUTER_ITS] RPGD_SHARED;
static int g_solver_in_use;
#endif
static RpgdWorkerScratch g_local_scratch RPGD_ALIGN64;
static float g_fd_q[RPGD_MAX_HORIZON] RPGD_ALIGN64;
static float g_ga[RPGD_MAX_HORIZON] RPGD_ALIGN64;
static float g_gf[RPGD_MAX_HORIZON] RPGD_ALIGN64;
#else
static RPGD_THREAD_LOCAL RpgdWorkerScratch tls_scratch;
static RPGD_THREAD_LOCAL float tls_fd_q[RPGD_MAX_HORIZON];
#endif

static float clampf_local(float x, float lo, float hi)
{
    if (!isfinite(x)) return 0.0f;
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static int finite_state6(const float* state6)
{
    if (!state6) return 0;
    for (int i = 0; i < STATE_DIM; ++i) {
        if (!isfinite(state6[i])) return 0;
    }
    return 1;
}

static int finite_runtime(const RpgdRuntime* rt)
{
    return rt
        && isfinite(rt->target_position)
        && isfinite(rt->target_equilibrium)
        && isfinite(rt->L)
        && isfinite(rt->m_pole)
        && rt->L >= 0.0f
        && rt->m_pole >= 0.0f;
}

static void normalize_config(RpgdConfig* dst, const RpgdConfig* src)
{
    *dst = *src;
    if (dst->period_interpolation_inducing_points <= 0) dst->period_interpolation_inducing_points = 1;
    if (dst->intermediate_steps <= 0) dst->intermediate_steps = 10;
    if (dst->resamp_per <= 0) dst->resamp_per = 1;
    if (dst->shift_previous <= 0) dst->shift_previous = 1;
    if (dst->reserve_threads < 0) dst->reserve_threads = 0;
}

static int validate_normalized_config(const RpgdConfig* c)
{
    if (c->mpc_horizon <= 0 || c->num_rollouts <= 0 || c->outer_its <= 0
        || c->intermediate_steps <= 0 || c->resamp_per <= 0
        || c->period_interpolation_inducing_points <= 0
        || c->shift_previous <= 0 || c->shift_previous > c->mpc_horizon
        || c->sampling_distribution < 0 || c->sampling_distribution > 1
        || c->sample_whole_control_space < 0 || c->sample_whole_control_space > 1
        || c->warmup < 0 || c->warmup > 1
        || (c->warmup && c->warmup_iterations <= 0)
        || c->num_threads < 0 || c->reserve_threads < 0) {
        return RPGD_STATUS_INVALID_CONFIG;
    }
    if (c->mpc_horizon > INT_MAX / c->intermediate_steps
        || (size_t)c->num_rollouts > SIZE_MAX / (size_t)c->mpc_horizon / sizeof(float)) {
        return RPGD_STATUS_INVALID_CONFIG;
    }
#define RPGD_FINITE_FIELD(field) if (!isfinite(c->field)) return RPGD_STATUS_INVALID_CONFIG
    RPGD_FINITE_FIELD(mpc_timestep);
    RPGD_FINITE_FIELD(learning_rate);
    RPGD_FINITE_FIELD(adam_beta_1);
    RPGD_FINITE_FIELD(adam_beta_2);
    RPGD_FINITE_FIELD(adam_epsilon);
    RPGD_FINITE_FIELD(gradmax_clip);
    RPGD_FINITE_FIELD(opt_keep_k_ratio);
    RPGD_FINITE_FIELD(sample_stdev);
    RPGD_FINITE_FIELD(sample_mean);
    RPGD_FINITE_FIELD(uniform_dist_min);
    RPGD_FINITE_FIELD(uniform_dist_max);
    RPGD_FINITE_FIELD(action_low);
    RPGD_FINITE_FIELD(action_high);
    RPGD_FINITE_FIELD(k);
    RPGD_FINITE_FIELD(m_cart);
    RPGD_FINITE_FIELD(m_pole);
    RPGD_FINITE_FIELD(g);
    RPGD_FINITE_FIELD(J_fric);
    RPGD_FINITE_FIELD(M_fric);
    RPGD_FINITE_FIELD(L);
    RPGD_FINITE_FIELD(u_max);
    RPGD_FINITE_FIELD(track_half_length);
    RPGD_FINITE_FIELD(dd_quadratic_weight_up);
    RPGD_FINITE_FIELD(db_weight_up);
    RPGD_FINITE_FIELD(ep_weight_up);
    RPGD_FINITE_FIELD(ekp_weight_up);
    RPGD_FINITE_FIELD(cc_weight_up);
    RPGD_FINITE_FIELD(vel_penalty_reg);
    RPGD_FINITE_FIELD(R);
    RPGD_FINITE_FIELD(permissible_track_fraction);
#undef RPGD_FINITE_FIELD
    if (c->mpc_timestep <= 0.0f || c->learning_rate < 0.0f
        || c->adam_beta_1 < 0.0f || c->adam_beta_1 >= 1.0f
        || c->adam_beta_2 < 0.0f || c->adam_beta_2 >= 1.0f
        || c->adam_epsilon <= 0.0f || c->gradmax_clip < 0.0f
        || c->opt_keep_k_ratio <= 0.0f || c->opt_keep_k_ratio > 1.0f
        || c->sample_stdev < 0.0f || c->uniform_dist_min > c->uniform_dist_max
        || c->action_low >= c->action_high || c->k <= -1.0f
        || c->m_cart <= 0.0f || c->m_pole <= 0.0f || c->g <= 0.0f
        || c->J_fric < 0.0f || c->M_fric < 0.0f || c->L <= 0.0f
        || c->u_max <= 0.0f || c->track_half_length <= 0.0f
        || c->permissible_track_fraction <= 0.0f
        || c->permissible_track_fraction >= 1.0f) {
        return RPGD_STATUS_INVALID_CONFIG;
    }
    return RPGD_STATUS_OK;
}

int rpgd_validate_config(const RpgdConfig* cfg)
{
    if (!cfg) return RPGD_STATUS_INVALID_ARGUMENT;
    RpgdConfig normalized;
    normalize_config(&normalized, cfg);
    return validate_normalized_config(&normalized);
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

#ifdef RPGD_PLATFORM_BAREMETAL
static int cfg_fits_static(const RpgdConfig* c)
{
    return c->num_rollouts <= RPGD_MAX_NUM_ROLLOUTS
        && c->mpc_horizon <= RPGD_MAX_HORIZON
        && c->intermediate_steps <= RPGD_MAX_INTERMEDIATE_STEPS
        && c->outer_its <= RPGD_MAX_OUTER_ITS
        && (!c->warmup || c->warmup_iterations <= RPGD_MAX_OUTER_ITS);
}
#endif

#ifndef RPGD_PLATFORM_BAREMETAL
static int state_scratch_fits(const RpgdConfig* c)
{
    return c->mpc_horizon <= RPGD_MAX_HORIZON
        && c->intermediate_steps <= RPGD_MAX_INTERMEDIATE_STEPS;
}
#endif

static void sample_action_sequence(RpgdSolver* solver, float* out_q)
{
    const RpgdConfig* c = &solver->cfg;
    const int h = c->mpc_horizon;
    const int p = c->period_interpolation_inducing_points;
    const int n = solver->inducing_points;
    float* points = solver->inducing;

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

static float rollout_cost(const RpgdConfig* c, const RpgdRuntime* rt, const float* state6, const float* q)
{
    float s[STATE_DIM];
    float sn[STATE_DIM];
    const float dt = c->mpc_timestep / (float)c->intermediate_steps;
    memcpy(s, state6, STATE_DIM * sizeof(float));

    float total = 0.0f;
    for (int h = 0; h < c->mpc_horizon; ++h) {
        total += cartpole_cost_stage(c, rt, s, q[h]);
        for (int k = 0; k < c->intermediate_steps; ++k) {
            cartpole_model_substep_dt(c, rt, s, q[h], dt, sn);
            memcpy(s, sn, STATE_DIM * sizeof(float));
        }
    }
    total /= (float)(c->mpc_horizon + 1);
    return isfinite(total) ? total : INFINITY;
}

static void rollout_gradient_fd_eps(
    const RpgdConfig* c,
    const RpgdRuntime* rt,
    const float* state6,
    const float* q,
    float eps,
    float* qp,
    float* grad
)
{
    memcpy(qp, q, (size_t)c->mpc_horizon * sizeof(float));
    for (int h = 0; h < c->mpc_horizon; ++h) {
        const float old = qp[h];
        const float plus = clampf_local(old + eps, c->action_low, c->action_high);
        const float minus = clampf_local(old - eps, c->action_low, c->action_high);
        qp[h] = plus;
        const float cp = rollout_cost(c, rt, state6, qp);
        qp[h] = minus;
        const float cm = rollout_cost(c, rt, state6, qp);
        qp[h] = old;
        const float effective_span = plus - minus;
        grad[h] = effective_span > 0.0f ? (cp - cm) / effective_span : 0.0f;
    }
}

static void rollout_gradient_adjoint_ws(
    const RpgdConfig* c,
    const RpgdRuntime* rt,
    const float* restrict state6,
    const float* restrict q,
    float* restrict states,
    float* restrict grad
) RPGD_HOT;

static void rollout_gradient_adjoint_ws(
    const RpgdConfig* c,
    const RpgdRuntime* rt,
    const float* restrict state6,
    const float* restrict q,
    float* restrict states,
    float* restrict grad
)
{
    const int H = c->mpc_horizon;
    const int K = c->intermediate_steps;
    const int total_steps = H * K;
    const float dt = c->mpc_timestep / (float)c->intermediate_steps;
    const float scale = 1.0f / (float)(H + 1);

    memset(grad, 0, (size_t)H * sizeof(float));
    memcpy(states, state6, STATE_DIM * sizeof(float));
    for (int n = 0; n < total_steps; ++n) {
        const int h = n / K;
        cartpole_model_substep_dt(c, rt, &states[(size_t)n * STATE_DIM], q[h], dt,
                                  &states[(size_t)(n + 1) * STATE_DIM]);
    }

    for (int h = 0; h < H; ++h) {
        grad[h] = cartpole_cost_stage_grad_q(c, q[h], scale);
    }

    float a_next[STATE_DIM];
    memset(a_next, 0, sizeof(a_next));
    for (int n = total_steps - 1; n >= 0; --n) {
        const int h = n / K;
        float a_cur[STATE_DIM];
        memset(a_cur, 0, sizeof(a_cur));
        if ((n % K) == 0) {
            cartpole_cost_stage_grad_state(
                c, rt, &states[(size_t)n * STATE_DIM], scale, a_cur);
        }
        float Jx[STATE_DIM][STATE_DIM];
        float Ju[STATE_DIM];
        const float* s_next = &states[(size_t)(n + 1) * STATE_DIM];
        cartpole_model_substep_jacobian_with_trig(
            c, rt, &states[(size_t)n * STATE_DIM], q[h], dt,
            s_next[ANGLE_COS_IDX], s_next[ANGLE_SIN_IDX], Jx, Ju);
        for (int j = 0; j < STATE_DIM; ++j) {
            float acc = 0.0f;
            for (int i = 0; i < STATE_DIM; ++i) acc += a_next[i] * Jx[i][j];
            a_cur[j] += acc;
        }
        float gu = 0.0f;
        for (int i = 0; i < STATE_DIM; ++i) gu += a_next[i] * Ju[i];
        grad[h] += gu;
        memcpy(a_next, a_cur, sizeof(a_next));
    }
}

static RpgdWorkerScratch* default_scratch(void)
{
#ifdef RPGD_PLATFORM_BAREMETAL
    return &g_local_scratch;
#else
    return &tls_scratch;
#endif
}

static float* scratch_states(const RpgdConfig* c)
{
#ifdef RPGD_PLATFORM_BAREMETAL
    (void)c;
    return g_local_scratch.states;
#else
    if (state_scratch_fits(c)) return tls_scratch.states;
    return NULL;
#endif
}

#ifndef RPGD_PLATFORM_BAREMETAL
static float* scratch_grad(const RpgdConfig* c)
{
    if (c->mpc_horizon <= RPGD_MAX_HORIZON) return tls_scratch.grad;
    return NULL;
}
#endif

static void rollout_gradient_adjoint(
    const RpgdConfig* c,
    const RpgdRuntime* rt,
    const float* state6,
    const float* q,
    float* grad
)
{
    float* states = scratch_states(c);
#ifndef RPGD_PLATFORM_BAREMETAL
    float* heap_states = NULL;
    if (!states) {
        heap_states = (float*)calloc((size_t)(c->mpc_horizon * c->intermediate_steps + 1) * STATE_DIM, sizeof(float));
        states = heap_states;
    }
    if (!states) return;
#endif
    rollout_gradient_adjoint_ws(c, rt, state6, q, states, grad);
#ifndef RPGD_PLATFORM_BAREMETAL
    free(heap_states);
#endif
}

static void rollout_gradient_fd(const RpgdConfig* c, const RpgdRuntime* rt, const float* state6, const float* q, float* grad)
{
#ifdef RPGD_PLATFORM_BAREMETAL
    rollout_gradient_fd_eps(c, rt, state6, q, 1.0e-2f, g_fd_q, grad);
#else
    float* qp = (c->mpc_horizon <= RPGD_MAX_HORIZON)
        ? tls_fd_q
        : (float*)malloc((size_t)c->mpc_horizon * sizeof(float));
    if (!qp) return;
    rollout_gradient_fd_eps(c, rt, state6, q, 1.0e-2f, qp, grad);
    if (qp != tls_fd_q) free(qp);
#endif
}

static int clip_gradient_norm(const RpgdConfig* c, float* grad)
{
    float sum = 0.0f;
    for (int h = 0; h < c->mpc_horizon; ++h) {
        if (!isfinite(grad[h])) return 0;
        sum += grad[h] * grad[h];
    }
    const float norm = sqrtf(sum);
    if (!isfinite(norm)) return 0;
    if (norm > c->gradmax_clip && norm > 0.0f) {
        const float scale = c->gradmax_clip / norm;
        for (int h = 0; h < c->mpc_horizon; ++h) grad[h] *= scale;
    }
    return 1;
}

static int adam_update_rollout(
    RpgdSolver* solver,
    int rollout,
    const float* grad,
    int iteration
)
{
    RpgdConfig* c = &solver->cfg;
    const int H = c->mpc_horizon;
    float* restrict q = &solver->q[(size_t)rollout * H];
    float* restrict m = &solver->adam_m[(size_t)rollout * H];
    float* restrict v = &solver->adam_v[(size_t)rollout * H];
    const float bc1 = solver->bias_correction_1[iteration];
    const float bc2 = solver->bias_correction_2[iteration];
    const float om1 = solver->one_minus_beta1;
    const float om2 = solver->one_minus_beta2;
    for (int h = 0; h < H; ++h) {
        m[h] = c->adam_beta_1 * m[h] + om1 * grad[h];
        v[h] = c->adam_beta_2 * v[h] + om2 * grad[h] * grad[h];
        const float m_hat = m[h] / bc1;
        const float v_hat = v[h] / bc2;
        const float updated =
            q[h] - c->learning_rate * m_hat / (sqrtf(v_hat) + c->adam_epsilon);
        if (!isfinite(updated)) return 0;
        q[h] = clampf_local(updated, c->action_low, c->action_high);
        if (!isfinite(q[h]) || !isfinite(m[h]) || !isfinite(v[h])) return 0;
    }
    return 1;
}

static int optimize_rollout_ws(
    RpgdSolver* solver,
    const RpgdRuntime* rt,
    const float* state6,
    int rollout,
    float* states,
    float* grad
)
{
    float* q = &solver->q[(size_t)rollout * solver->cfg.mpc_horizon];
    if (!states || !grad) {
        solver->costs[rollout] = INFINITY;
        return 0;
    }
    for (int it = 0; it < solver->active_iterations; ++it) {
        rollout_gradient_adjoint_ws(&solver->cfg, rt, state6, q, states, grad);
        if (!clip_gradient_norm(&solver->cfg, grad)
            || !adam_update_rollout(solver, rollout, grad, it)) {
            solver->costs[rollout] = INFINITY;
            return 0;
        }
    }
    solver->costs[rollout] = rollout_cost(&solver->cfg, rt, state6, q);
    return isfinite(solver->costs[rollout]);
}

#ifndef RPGD_PLATFORM_BAREMETAL
static int optimize_rollout(RpgdSolver* solver, const RpgdRuntime* rt, const float* state6, int rollout)
{
    const int H = solver->cfg.mpc_horizon;
    float* grad = scratch_grad(&solver->cfg);
    float* states = scratch_states(&solver->cfg);
#ifndef RPGD_PLATFORM_BAREMETAL
    float* heap_grad = NULL;
    float* heap_states = NULL;
    if (!grad) {
        heap_grad = (float*)malloc((size_t)H * sizeof(float));
        grad = heap_grad;
    }
    if (!states) {
        heap_states = (float*)calloc(
            (size_t)(solver->cfg.mpc_horizon * solver->cfg.intermediate_steps + 1) * STATE_DIM,
            sizeof(float));
        states = heap_states;
    }
    if (!grad || !states) {
        free(heap_grad);
        free(heap_states);
        solver->costs[rollout] = INFINITY;
        return 0;
    }
#else
    (void)H;
#endif
    const int ok = optimize_rollout_ws(solver, rt, state6, rollout, states, grad);
#ifndef RPGD_PLATFORM_BAREMETAL
    free(heap_grad);
    free(heap_states);
#endif
    return ok;
}
#endif /* !RPGD_PLATFORM_BAREMETAL */

#if !defined(RPGD_PLATFORM_BAREMETAL) && !defined(_OPENMP)
static void* optimize_rollout_range(void* arg)
{
    RpgdThreadArgs* a = (RpgdThreadArgs*)arg;
    for (int i = a->start; i < a->end; ++i) {
        optimize_rollout(a->solver, a->runtime, a->state6, i);
    }
    return NULL;
}
#endif

static void shift_keep(
    RpgdSolver* solver,
    int dst,
    int src
)
{
    const RpgdConfig* c = &solver->cfg;
    const int H = c->mpc_horizon;
    float* dq = &solver->warm_q[(size_t)dst * H];
    float* dm = &solver->warm_m[(size_t)dst * H];
    float* dv = &solver->warm_v[(size_t)dst * H];
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
}

static void warmstart_after_action(RpgdSolver* solver)
{
    RpgdConfig* c = &solver->cfg;
    const int N = c->num_rollouts;
    const int H = c->mpc_horizon;
    const int keep = solver->opt_keep_k;
    const int do_resample = solver->resample_phase == 0;
    float* qn = solver->warm_q;
    float* mn = solver->warm_m;
    float* vn = solver->warm_v;
    float* ages = solver->warm_ages;
    const size_t q_bytes = (size_t)N * (size_t)H * sizeof(float);

    int dst = 0;
    if (do_resample) {
        const int sampled = N - keep;
        memset(mn, 0, (size_t)sampled * (size_t)H * sizeof(float));
        memset(vn, 0, (size_t)sampled * (size_t)H * sizeof(float));
        for (; dst < sampled; ++dst) {
            sample_action_sequence(solver, &qn[(size_t)dst * H]);
            ages[dst] = 0.0f;
        }
        for (int k = 0; k < keep; ++k, ++dst) {
            const int src = solver->indices[k];
            shift_keep(solver, dst, src);
            ages[dst] = solver->trajectory_ages[src];
        }
    } else {
        for (int src = 0; src < N; ++src) {
            shift_keep(solver, src, src);
            ages[src] = solver->trajectory_ages[src];
        }
    }
    for (int i = 0; i < N; ++i) ages[i] += 1.0f;
    memcpy(solver->q, qn, q_bytes);
    memcpy(solver->adam_m, mn, q_bytes);
    memcpy(solver->adam_v, vn, q_bytes);
    memcpy(solver->trajectory_ages, ages, (size_t)N * sizeof(float));
}

static int choose_thread_count(const RpgdConfig* cfg)
{
#ifdef RPGD_FORCE_SINGLE_THREAD
    (void)cfg;
    return 1;
#else
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
#endif
}

static void apply_cfg_defaults(RpgdSolver* solver, const RpgdConfig* cfg)
{
    solver->cfg = *cfg;
    solver->opt_keep_k = (int)(cfg->num_rollouts * cfg->opt_keep_k_ratio);
    if (solver->opt_keep_k < 1) solver->opt_keep_k = 1;
    if (solver->opt_keep_k > cfg->num_rollouts) solver->opt_keep_k = cfg->num_rollouts;
    solver->inducing_points = calc_inducing_points(cfg->mpc_horizon, solver->cfg.period_interpolation_inducing_points);
    solver->thread_count = choose_thread_count(&solver->cfg);
    solver->one_minus_beta1 = 1.0f - solver->cfg.adam_beta_1;
    solver->one_minus_beta2 = 1.0f - solver->cfg.adam_beta_2;
    solver->last_status = RPGD_STATUS_OK;
    solver->busy = 0;
    solver->active_iterations = solver->cfg.outer_its;
}

RpgdSolver* rpgd_create(const RpgdConfig* cfg)
{
#ifdef RPGD_WORKER_ONLY
    (void)cfg;
    return NULL;
#else
    if (!cfg) return NULL;
    RpgdConfig normalized;
    normalize_config(&normalized, cfg);
    if (validate_normalized_config(&normalized) != RPGD_STATUS_OK) return NULL;
#ifdef RPGD_PLATFORM_BAREMETAL
    if (!cfg_fits_static(&normalized) || g_solver_in_use) return NULL;
    memset(&g_solver, 0, sizeof(g_solver));
    RpgdSolver* solver = &g_solver;
    apply_cfg_defaults(solver, &normalized);
    solver->q = g_q;
    solver->adam_m = g_adam_m;
    solver->adam_v = g_adam_v;
    solver->trajectory_ages = g_trajectory_ages;
    solver->costs = g_costs;
    solver->indices = g_indices;
    solver->inducing = g_inducing;
    solver->warm_q = g_warm_q;
    solver->warm_m = g_warm_m;
    solver->warm_v = g_warm_v;
    solver->warm_ages = g_warm_ages;
    solver->bias_correction_1 = g_bias_correction_1;
    solver->bias_correction_2 = g_bias_correction_2;
    solver->owns_storage = 0;
    solver->busy = 0;
    solver->workspace_bytes =
        sizeof(g_solver) + sizeof(g_q) + sizeof(g_adam_m) + sizeof(g_adam_v)
        + sizeof(g_trajectory_ages) + sizeof(g_costs) + sizeof(g_indices)
        + sizeof(g_inducing) + sizeof(g_warm_q) + sizeof(g_warm_m) + sizeof(g_warm_v)
        + sizeof(g_warm_ages) + sizeof(g_local_scratch)
        + sizeof(g_fd_q) + sizeof(g_ga) + sizeof(g_gf)
        + sizeof(g_bias_correction_1) + sizeof(g_bias_correction_2)
        + sizeof(g_solver_in_use);
    memset(g_q, 0, sizeof(g_q));
    memset(g_adam_m, 0, sizeof(g_adam_m));
    memset(g_adam_v, 0, sizeof(g_adam_v));
    memset(g_trajectory_ages, 0, sizeof(g_trajectory_ages));
    memset(g_costs, 0, sizeof(g_costs));
    memset(g_indices, 0, sizeof(g_indices));
    g_solver_in_use = 1;
    rpgd_reset(solver, normalized.seed);
    return solver;
#else
    RpgdSolver* solver = (RpgdSolver*)calloc(1, sizeof(RpgdSolver));
    if (!solver) return NULL;
    apply_cfg_defaults(solver, &normalized);
    const size_t q_size = (size_t)normalized.num_rollouts * (size_t)normalized.mpc_horizon;
    solver->q = (float*)calloc(q_size, sizeof(float));
    solver->adam_m = (float*)calloc(q_size, sizeof(float));
    solver->adam_v = (float*)calloc(q_size, sizeof(float));
    solver->trajectory_ages = (float*)calloc((size_t)normalized.num_rollouts, sizeof(float));
    solver->costs = (float*)calloc((size_t)normalized.num_rollouts, sizeof(float));
    solver->indices = (int*)calloc((size_t)normalized.num_rollouts, sizeof(int));
    solver->inducing = (float*)calloc((size_t)solver->inducing_points, sizeof(float));
    solver->warm_q = (float*)calloc(q_size, sizeof(float));
    solver->warm_m = (float*)calloc(q_size, sizeof(float));
    solver->warm_v = (float*)calloc(q_size, sizeof(float));
    solver->warm_ages = (float*)calloc((size_t)normalized.num_rollouts, sizeof(float));
    const int max_iterations =
        normalized.warmup && normalized.warmup_iterations > normalized.outer_its
        ? normalized.warmup_iterations
        : normalized.outer_its;
    solver->bias_correction_1 = (float*)calloc((size_t)max_iterations, sizeof(float));
    solver->bias_correction_2 = (float*)calloc((size_t)max_iterations, sizeof(float));
    solver->owns_storage = 1;
    solver->workspace_bytes =
        sizeof(RpgdSolver)
        + (3 * q_size + 2 * (size_t)normalized.num_rollouts + (size_t)solver->inducing_points
           + 3 * q_size + (size_t)normalized.num_rollouts + 2 * (size_t)max_iterations)
          * sizeof(float)
        + (size_t)normalized.num_rollouts * sizeof(int);
    if (!solver->q || !solver->adam_m || !solver->adam_v || !solver->trajectory_ages
        || !solver->costs || !solver->indices || !solver->inducing
        || !solver->warm_q || !solver->warm_m || !solver->warm_v || !solver->warm_ages
        || !solver->bias_correction_1 || !solver->bias_correction_2) {
        rpgd_destroy(solver);
        return NULL;
    }
    rpgd_reset(solver, normalized.seed);
    return solver;
#endif /* RPGD_PLATFORM_BAREMETAL */
#endif /* RPGD_WORKER_ONLY */
}

void rpgd_destroy(RpgdSolver* solver)
{
    if (!solver) return;
#ifdef RPGD_WORKER_ONLY
    (void)solver;
    return;
#elif defined(RPGD_PLATFORM_BAREMETAL)
    if (solver == &g_solver) g_solver_in_use = 0;
    solver->busy = 0;
    return;
#else
    if (solver->owns_storage) {
        free(solver->q);
        free(solver->adam_m);
        free(solver->adam_v);
        free(solver->trajectory_ages);
        free(solver->costs);
        free(solver->indices);
        free(solver->inducing);
        free(solver->warm_q);
        free(solver->warm_m);
        free(solver->warm_v);
        free(solver->warm_ages);
        free(solver->bias_correction_1);
        free(solver->bias_correction_2);
    }
    free(solver);
#endif
}

void rpgd_reset(RpgdSolver* solver, unsigned int seed)
{
    if (!solver) return;
    rng_seed(&solver->rng, seed ? seed : solver->cfg.seed);
    solver->resample_phase = 0;
    solver->first_step = 1;
    solver->adam_step = 0;
    solver->last_status = RPGD_STATUS_OK;
    solver->busy = 0;
    const int N = solver->cfg.num_rollouts;
    const int H = solver->cfg.mpc_horizon;
    for (int i = 0; i < N; ++i) sample_action_sequence(solver, &solver->q[(size_t)i * H]);
    memset(solver->adam_m, 0, (size_t)N * H * sizeof(float));
    memset(solver->adam_v, 0, (size_t)N * H * sizeof(float));
    memset(solver->trajectory_ages, 0, (size_t)N * sizeof(float));
}

#if !defined(RPGD_PLATFORM_BAREMETAL) && !defined(RPGD_FORCE_SINGLE_THREAD)
static void optimize_all_rollouts(RpgdSolver* solver, const float* state6, const RpgdRuntime* runtime)
{
    const int N = solver->cfg.num_rollouts;
#if defined(RPGD_PLATFORM_BAREMETAL) || defined(RPGD_FORCE_SINGLE_THREAD)
    for (int i = 0; i < N; ++i) {
        optimize_rollout(solver, runtime, state6, i);
    }
#elif defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(solver->thread_count)
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
        unsigned char* started = (unsigned char*)calloc((size_t)solver->thread_count, sizeof(unsigned char));
        if (!threads || !args || !started) {
            free(threads);
            free(args);
            free(started);
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
                if (pthread_create(&threads[t], NULL, optimize_rollout_range, &args[t]) == 0) {
                    started[t] = 1;
                } else {
                    for (int i = start; i < end; ++i) {
                        optimize_rollout(solver, runtime, state6, i);
                    }
                    solver->last_status = RPGD_STATUS_THREAD_FAILURE;
                }
            }
            for (int t = 0; t < solver->thread_count; ++t) {
                if (started[t]) pthread_join(threads[t], NULL);
            }
            free(threads);
            free(args);
            free(started);
        }
    }
#endif
}
#endif /* host multi-thread optimize_all_rollouts */

float rpgd_step(RpgdSolver* solver, const float* state6, const RpgdRuntime* runtime)
{
    RpgdStepPlan plan;
    const int rc = rpgd_step_prepare(solver, state6, runtime, &plan);
    if (rc != RPGD_STATUS_OK) return 0.0f;
#if defined(RPGD_PLATFORM_BAREMETAL) || defined(RPGD_FORCE_SINGLE_THREAD)
    const int opt_rc = rpgd_step_optimize_range(
        solver, &plan, 0, solver->cfg.num_rollouts, default_scratch());
    if (opt_rc != RPGD_STATUS_OK) {
        rpgd_step_abort(solver, opt_rc);
        return 0.0f;
    }
#else
    optimize_all_rollouts(solver, plan.state6, &plan.runtime);
#endif
    return rpgd_step_finalize(solver, &plan);
}

int rpgd_step_prepare(RpgdSolver* solver, const float state6[6],
                      const RpgdRuntime* runtime, RpgdStepPlan* plan)
{
    if (!solver) return RPGD_STATUS_INVALID_ARGUMENT;
    if (solver->busy) {
        solver->last_status = RPGD_STATUS_BUSY;
        return RPGD_STATUS_BUSY;
    }
    if (!plan) {
        solver->last_status = RPGD_STATUS_INVALID_ARGUMENT;
        return RPGD_STATUS_INVALID_ARGUMENT;
    }
    memset(plan, 0, sizeof(*plan));
    solver->last_status = RPGD_STATUS_OK;
    if (!finite_state6(state6) || !finite_runtime(runtime)) {
        solver->last_status = RPGD_STATUS_INVALID_ARGUMENT;
        return RPGD_STATUS_INVALID_ARGUMENT;
    }
    solver->busy = 1;
    memcpy(plan->state6, state6, STATE_DIM * sizeof(float));
    plan->runtime = *runtime;
    solver->active_iterations =
        solver->first_step && solver->cfg.warmup
        ? solver->cfg.warmup_iterations
        : solver->cfg.outer_its;
    plan->active_iterations = solver->active_iterations;
    for (int it = 0; it < solver->active_iterations; ++it) {
        uint64_t step = solver->adam_step + (uint64_t)it + UINT64_C(1);
        const float step_f = step > UINT64_C(16777216) ? 16777216.0f : (float)step;
        solver->bias_correction_1[it] =
            1.0f - powf(solver->cfg.adam_beta_1, step_f);
        solver->bias_correction_2[it] =
            1.0f - powf(solver->cfg.adam_beta_2, step_f);
    }
    const int N = solver->cfg.num_rollouts;
    for (int i = 0; i < N; ++i) solver->costs[i] = INFINITY;
    plan->prepared = 1;
    plan->range_error = RPGD_STATUS_OK;
    return RPGD_STATUS_OK;
}

int rpgd_step_optimize_range(RpgdSolver* solver, const RpgdStepPlan* plan,
                             int first, int last,
                             RpgdWorkerScratch* scratch)
{
    RpgdStepPlan* mut = (RpgdStepPlan*)plan;
    if (!solver || !plan || !plan->prepared || !solver->busy) {
#ifndef RPGD_WORKER_ONLY
        if (solver) solver->last_status = RPGD_STATUS_INVALID_ARGUMENT;
#endif
        if (mut) mut->range_error = RPGD_STATUS_INVALID_ARGUMENT;
        return RPGD_STATUS_INVALID_ARGUMENT;
    }
    const int N = solver->cfg.num_rollouts;
    if (first < 0 || last < first || last > N) {
#ifndef RPGD_WORKER_ONLY
        solver->last_status = RPGD_STATUS_INVALID_ARGUMENT;
#endif
        mut->range_error = RPGD_STATUS_INVALID_ARGUMENT;
        return RPGD_STATUS_INVALID_ARGUMENT;
    }
    RpgdWorkerScratch* ws = scratch ? scratch : default_scratch();
#ifndef RPGD_PLATFORM_BAREMETAL
    if (!state_scratch_fits(&solver->cfg) || solver->cfg.mpc_horizon > RPGD_MAX_HORIZON) {
        for (int i = first; i < last; ++i) {
            optimize_rollout(solver, &plan->runtime, plan->state6, i);
        }
        return RPGD_STATUS_OK;
    }
#endif
    for (int i = first; i < last; ++i) {
        optimize_rollout_ws(solver, &plan->runtime, plan->state6, i, ws->states, ws->grad);
    }
    return RPGD_STATUS_OK;
}

void rpgd_step_abort(RpgdSolver* solver, int status)
{
    if (!solver) return;
    if (status != RPGD_STATUS_OK) solver->last_status = status;
    solver->busy = 0;
}

float rpgd_step_finalize(RpgdSolver* solver, RpgdStepPlan* plan)
{
    if (!solver) return 0.0f;
    if (!plan || !plan->prepared || !solver->busy) {
        solver->last_status = RPGD_STATUS_INVALID_ARGUMENT;
        solver->busy = 0;
        return 0.0f;
    }
    if (plan->range_error != RPGD_STATUS_OK) {
        rpgd_step_abort(solver, plan->range_error);
        plan->prepared = 0;
        return 0.0f;
    }
    if (solver->last_status == RPGD_STATUS_WORKER_FAILURE) {
        rpgd_step_abort(solver, RPGD_STATUS_WORKER_FAILURE);
        plan->prepared = 0;
        return 0.0f;
    }
    const int N = solver->cfg.num_rollouts;
    sort_indices_by_cost(solver->costs, solver->indices, N);
    if (!isfinite(solver->costs[solver->indices[0]])) {
        solver->last_status = RPGD_STATUS_NUMERICAL_FAILURE;
        solver->busy = 0;
        plan->prepared = 0;
        return 0.0f;
    }
    if (solver->adam_step <= UINT64_MAX - (uint64_t)solver->active_iterations) {
        solver->adam_step += (uint64_t)solver->active_iterations;
    } else {
        solver->adam_step = UINT64_MAX;
    }
    const float u = solver->q[(size_t)solver->indices[0] * solver->cfg.mpc_horizon];
    warmstart_after_action(solver);
    solver->first_step = 0;
    solver->resample_phase += 1;
    if (solver->resample_phase >= solver->cfg.resamp_per) solver->resample_phase = 0;
    solver->busy = 0;
    plan->prepared = 0;
    if (solver->last_status != RPGD_STATUS_THREAD_FAILURE) {
        solver->last_status = RPGD_STATUS_OK;
    }
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
#ifdef RPGD_PLATFORM_BAREMETAL
    if (cfg->mpc_horizon > RPGD_MAX_HORIZON) return -2;
    float* ga = g_ga;
    float* gf = g_gf;
#else
    float* ga = (float*)calloc((size_t)cfg->mpc_horizon, sizeof(float));
    float* gf = (float*)calloc((size_t)cfg->mpc_horizon, sizeof(float));
    if (!ga || !gf) {
        free(ga); free(gf);
        return -2;
    }
#endif
    rollout_gradient_adjoint(cfg, runtime, state6, q, ga);
#ifdef RPGD_PLATFORM_BAREMETAL
    rollout_gradient_fd_eps(cfg, runtime, state6, q, eps > 0.0f ? eps : 1.0e-2f, g_fd_q, gf);
#else
    float* qp = (cfg->mpc_horizon <= RPGD_MAX_HORIZON)
        ? tls_fd_q
        : (float*)malloc((size_t)cfg->mpc_horizon * sizeof(float));
    if (!qp) {
        free(ga); free(gf);
        return -2;
    }
    rollout_gradient_fd_eps(cfg, runtime, state6, q, eps > 0.0f ? eps : 1.0e-2f, qp, gf);
    if (qp != tls_fd_q) free(qp);
#endif
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
#ifndef RPGD_PLATFORM_BAREMETAL
    free(ga); free(gf);
#endif
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
    solver->adam_step = step > 0 ? (uint64_t)step : UINT64_C(0);
}

void rpgd_debug_get_adam(const RpgdSolver* solver, float* m, float* v, int* step)
{
    if (!solver) return;
    const size_t n = (size_t)solver->cfg.num_rollouts * solver->cfg.mpc_horizon;
    if (m) memcpy(m, solver->adam_m, n * sizeof(float));
    if (v) memcpy(v, solver->adam_v, n * sizeof(float));
    if (step) {
        *step = solver->adam_step > (uint64_t)INT_MAX
            ? INT_MAX
            : (int)solver->adam_step;
    }
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

size_t rpgd_get_workspace_bytes(const RpgdSolver* solver)
{
    return solver ? solver->workspace_bytes : 0;
}

size_t rpgd_get_static_workspace_bytes(void)
{
    return sizeof(RpgdSolver)
        + (size_t)(6 * RPGD_MAX_Q_BUF) * sizeof(float)
        + (size_t)(3 * RPGD_MAX_NUM_ROLLOUTS) * sizeof(float)
        + (size_t)RPGD_MAX_NUM_ROLLOUTS * sizeof(int)
        + (size_t)RPGD_MAX_HORIZON * sizeof(float)
        + sizeof(RpgdWorkerScratch)
        + (size_t)(3 * RPGD_MAX_HORIZON) * sizeof(float)
        + (size_t)(2 * RPGD_MAX_OUTER_ITS) * sizeof(float)
        + sizeof(int);
}

int rpgd_get_last_status(const RpgdSolver* solver)
{
    return solver ? solver->last_status : RPGD_STATUS_INVALID_ARGUMENT;
}

int rpgd_is_busy(const RpgdSolver* solver)
{
    return solver ? solver->busy : 0;
}

int rpgd_get_resample_phase(const RpgdSolver* solver)
{
    return solver ? solver->resample_phase : 0;
}

int rpgd_is_baremetal(void)
{
#ifdef RPGD_PLATFORM_BAREMETAL
    return 1;
#else
    return 0;
#endif
}

unsigned int rpgd_get_abi_version(void)
{
    return RPGD_ABI_VERSION;
}

size_t rpgd_get_config_size(void)
{
    return sizeof(RpgdConfig);
}

size_t rpgd_get_solver_size(void)
{
    return sizeof(RpgdSolver);
}

size_t rpgd_get_worker_scratch_bytes(void)
{
    return sizeof(RpgdWorkerScratch);
}

void rpgd_cache_visit_solver(RpgdSolver* solver, void (*fn)(const void*, size_t))
{
    if (!solver || !fn) return;
    fn(solver, sizeof(*solver));
    if (solver->bias_correction_1 && solver->active_iterations > 0) {
        fn(solver->bias_correction_1, (size_t)solver->active_iterations * sizeof(float));
        fn(solver->bias_correction_2, (size_t)solver->active_iterations * sizeof(float));
    }
    if (solver->costs) {
        fn(solver->costs, (size_t)solver->cfg.num_rollouts * sizeof(float));
    }
}

void rpgd_cache_visit_rollout_slice(RpgdSolver* solver, int first, int last,
                                    void (*fn)(const void*, size_t))
{
    if (!solver || !fn) return;
    const int N = solver->cfg.num_rollouts;
    const int H = solver->cfg.mpc_horizon;
    if (first < 0) first = 0;
    if (last > N) last = N;
    if (last <= first || H <= 0) return;
    const size_t q_bytes = (size_t)(last - first) * (size_t)H * sizeof(float);
    fn(&solver->q[(size_t)first * (size_t)H], q_bytes);
    fn(&solver->adam_m[(size_t)first * (size_t)H], q_bytes);
    fn(&solver->adam_v[(size_t)first * (size_t)H], q_bytes);
    fn(&solver->costs[first], (size_t)(last - first) * sizeof(float));
}
