#ifndef RPGD_CARTPOLE_H
#define RPGD_CARTPOLE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int mpc_horizon;
    int num_rollouts;
    int outer_its;
    int resamp_per;
    int period_interpolation_inducing_points;
    int intermediate_steps;
    int shift_previous;
    int sampling_distribution; /* 0: normal, 1: uniform */
    int sample_whole_control_space;
    int warmup;
    int warmup_iterations;
    int num_threads;       /* 0: auto */
    int reserve_threads;   /* used when num_threads == 0 */
    unsigned int seed;

    float mpc_timestep;
    float learning_rate;
    float adam_beta_1;
    float adam_beta_2;
    float adam_epsilon;
    float gradmax_clip;
    float opt_keep_k_ratio;
    float sample_stdev;
    float sample_mean;
    float uniform_dist_min;
    float uniform_dist_max;
    float action_low;
    float action_high;

    float k;
    float m_cart;
    float m_pole;
    float g;
    float J_fric;
    float M_fric;
    float L;
    float u_max;
    float track_half_length;

    float dd_quadratic_weight_up;
    float db_weight_up;
    float ep_weight_up;
    float ekp_weight_up;
    float cc_weight_up;
    float vel_penalty_reg;
    float R;
    float permissible_track_fraction;
} RpgdConfig;

typedef struct {
    float target_position;
    float target_equilibrium;
    float L;
    float m_pole;
} RpgdRuntime;

typedef struct RpgdSolver RpgdSolver;

enum {
    RPGD_STATUS_OK = 0,
    RPGD_STATUS_INVALID_ARGUMENT = -1,
    RPGD_STATUS_INVALID_CONFIG = -2,
    RPGD_STATUS_BUSY = -3,
    RPGD_STATUS_WORKSPACE_FAILURE = -4,
    RPGD_STATUS_THREAD_FAILURE = -5,
    RPGD_STATUS_NUMERICAL_FAILURE = -6
};

int rpgd_validate_config(const RpgdConfig* cfg);
RpgdSolver* rpgd_create(const RpgdConfig* cfg);
void rpgd_destroy(RpgdSolver* solver);
void rpgd_reset(RpgdSolver* solver, unsigned int seed);
float rpgd_step(RpgdSolver* solver, const float* state6, const RpgdRuntime* runtime);

int rpgd_gradient_check(
    const RpgdConfig* cfg,
    const RpgdRuntime* runtime,
    const float* state6,
    const float* q,
    float eps,
    float* max_abs_error,
    float* max_rel_error
);

float rpgd_debug_rollout_cost(
    const RpgdConfig* cfg,
    const RpgdRuntime* runtime,
    const float* state6,
    const float* q
);
void rpgd_debug_rollout_final_state(
    const RpgdConfig* cfg,
    const RpgdRuntime* runtime,
    const float* state6,
    const float* q,
    float* final_state6
);
void rpgd_debug_gradient_adjoint(
    const RpgdConfig* cfg,
    const RpgdRuntime* runtime,
    const float* state6,
    const float* q,
    float* grad
);
void rpgd_debug_gradient_fd(
    const RpgdConfig* cfg,
    const RpgdRuntime* runtime,
    const float* state6,
    const float* q,
    float* grad
);
void rpgd_debug_set_q(RpgdSolver* solver, const float* q);
void rpgd_debug_get_q(const RpgdSolver* solver, float* q);
void rpgd_debug_set_adam(RpgdSolver* solver, const float* m, const float* v, int step);
void rpgd_debug_get_adam(const RpgdSolver* solver, float* m, float* v, int* step);
void rpgd_debug_get_costs(const RpgdSolver* solver, float* costs);
void rpgd_debug_get_indices(const RpgdSolver* solver, int* indices);

int rpgd_get_num_threads(const RpgdSolver* solver);
int rpgd_get_num_rollouts(const RpgdSolver* solver);
int rpgd_get_horizon(const RpgdSolver* solver);
size_t rpgd_get_workspace_bytes(const RpgdSolver* solver);
size_t rpgd_get_static_workspace_bytes(void);
int rpgd_get_last_status(const RpgdSolver* solver);
int rpgd_is_baremetal(void);
unsigned int rpgd_get_abi_version(void);
size_t rpgd_get_config_size(void);

#ifdef __cplusplus
}
#endif

#endif
