For the PFI RPGD optimizer:

- To change collision reset of car, go to "gym/f110_gym/envs/base_classes.py" and edit from line 246 in method "check_ttc".
```python
    def check_ttc(self, current_scan):
        ...
        
        in_collision = check_ttc_jit(current_scan, self.state[3], self.scan_angles, self.cosines, self.side_distances, self.ttc_thresh)

        # if in collision stop vehicle
        if in_collision:
            # self.state[3:] = 0.  # Why would the angle be reset??????
            self.state[0] -= np.sign(self.state[3]) * np.cos(self.state[4]) / 10
            self.state[1] -= np.sign(self.state[3]) * np.sin(self.state[4]) / 10

            self.state[2:4] = 0.
            self.state[5:7] = 0.

            """self.state[0] -= self.state[3] * np.cos(self.state[4])
            self.state[1] -= self.state[3] * np.sin(self.state[4])"""

            """if self.state[4] >= np.pi:
                self.state[4] -= np.pi
            else:
                self.state[4] += np.pi"""

            self.accel = 0.0
            self.steer_angle_vel = 0.0

        # update state
        self.in_collision = in_collision

        return in_collision
```
- To change max and min allowed control inputs, go to "utilities/state_utilities.py" and change line 113
```python
if Settings.ENVIRONMENT_NAME == 'Car':
    if not Settings.WITH_PID:  # MPC return velocity and steering angle
        control_limits_low, control_limits_high = get_control_limits([[-3.2, -9.5], [3.2, 9.5]])
    else:  # MPC returns acceleration and steering velocity
        control_limits_low, control_limits_high = get_control_limits([[-1.066, -5], [1.066, 8]]) # <-HERE
else:
    raise NotImplementedError('{} mpc not implemented yet'.format(Settings.ENVIRONMENT_NAME))
```
- Deprecated code:
```python
@CompileTF
def tf_interp(x, xs, ys):
    # Normalize data types
    ys = tf.cast(ys, tf.float32)
    xs = tf.cast(xs, tf.float32)
    x = tf.cast(x, tf.float32)

    # Pad control points for extrapolation
    xs = tf.concat([[xs[0]], xs, [xs[-1]]], axis=0)
    ys = tf.concat([ys[:1], ys, ys[-1:]], axis=0)

    # Compute slopes, pad at the edges to flatten
    ms = (ys[1:] - ys[:-1]) / (xs[1:] - xs[:-1])
    ms = tf.pad(ms[:-1], [(1, 1)])

    # Solve for intercepts
    bs = ys - ms * xs

    # Find the line parameters at each input data point using searchsorted
    i = tf.searchsorted(xs, x)
    m = tf.gather(ms, i)
    b = tf.gather(bs, i)

    # Apply the linear mapping at each input data point
    y = m * x + b
    return y


@CompileTF
def get_interp_pdf_cdf(x_val, x_min, x_max, y_val, num_interp_pts):
    # Sort the x and y using TensorFlow operations
    x_sorted_indices = tf.argsort(x_val)
    x_sorted = tf.gather(x_val, x_sorted_indices)
    y_sorted = tf.gather(y_val, x_sorted_indices)

    # Linear interpolation using tf_interp
    x_interp = tf.linspace(x_min, x_max, num_interp_pts)
    y_interp = tf_interp(x_interp, x_sorted, y_sorted)

    # pdf using TensorFlow operations
    y_pdf = y_interp / tf.reduce_sum(y_interp)

    # cdf using TensorFlow operations
    y_cdf = tf.math.cumsum(y_pdf)

    return x_interp, y_pdf, y_cdf


def get_kpf_samples(self, Qn, uniform_samples_all):
    # return self.sample_actions(self.rng, self.kpf_num_resample)    # QUICKLY BYPASS KPF RESAMPLING
    weights = self.kpf_weights
    all_cis = []
    for i in range(self.num_control_inputs):
        all_timesteps = []
        for j in range(self.mpc_horizon):
            # Sort the x and y using TensorFlow operations
            (
                sample_x,
                _,
                cdf_y
            ) = get_interp_pdf_cdf(Qn[:, j, i],
                                   self.action_low[i],
                                   self.action_high[i],
                                   self.kpf_weights,
                                   self.kpf_cdf_interp_num)
            rev_sampled_input = tf_interp(uniform_samples_all[:, j, i], cdf_y, sample_x)

            all_timesteps.append(rev_sampled_input)
        all_cis.append(tf.stack(all_timesteps, axis=1))
    Qres = tf.stack(all_cis, axis=2)
    return Qres


@CompileTF
def get_kpf_samples(self, Qn, uniform_samples_all):
    # return self.sample_actions(self.rng, self.kpf_num_resample)    # QUICKLY BYPASS KPF RESAMPLING
    all_cis = []
    for i in range(self.num_control_inputs):
        all_timesteps = []
        Qn_add = tf.concat([tf.ones(shape=(1, self.mpc_horizon)) * self.action_low[i],
                            Qn[:, :, i],
                            tf.ones(shape=(1, self.mpc_horizon)) * self.action_high[i]],
                           axis=0)
        weights = self.kpf_weights
        weights = tf.concat([tf.reduce_max(weights, keepdims=True),
                             weights,
                             tf.reduce_max(weights, keepdims=True)],
                            axis=0)

        for j in range(self.mpc_horizon):
            # Sort the x and y using TensorFlow operations
            sliced_inputs = Qn_add[:, j]

            sorted_indices = tf.argsort(sliced_inputs)
            Qn_sorted = tf.gather(sliced_inputs, sorted_indices)
            weights_sorted = tf.gather(weights, sorted_indices)
            weights_cdf = tf.cumsum(weights_sorted)

            uniform_samples = uniform_samples_all[:, j, i]

            # Find indices for the sampled points using binary search
            indices = tf.math.minimum(tf.searchsorted(weights_cdf, uniform_samples), len(weights) - 1)

            # n1 = uniform_samples
            # n2 = tf.gather(weights_cdf, tf.math.maximum(indices - 1, 0))
            # n3 = tf.gather(weights_cdf, indices)

            prev_indices = tf.math.maximum(indices - 1, 0)

            alphas = tf.math.maximum(tf.math.divide(uniform_samples - tf.gather(weights_cdf, prev_indices),
                                                    tf.gather(weights_cdf, indices) - tf.gather(weights_cdf,
                                                                                                prev_indices))
                                     , 0)

            samples_per_ci_per_timestep = tf.gather(Qn_sorted, prev_indices) + alphas * (
                    tf.gather(Qn_sorted, indices) - tf.gather(Qn_sorted, prev_indices))

            all_timesteps.append(samples_per_ci_per_timestep)
        all_cis.append(tf.stack(all_timesteps, axis=1))
    Qres = tf.stack(all_cis, axis=2)
    return Qres

#IN STEP:
"""uniform_samples = tf.random.uniform(shape=(self.kpf_num_resample,
                                                                   self.mpc_horizon,
                                                                   self.num_control_inputs))  # possibility to define it once instead of each time to save time"""

"""uniform_samples = tf.tile(tf.expand_dims(tf.linspace(0.0, 1.0, self.kpf_num_resample), axis=0),
                                      num_copies, 1, 1])"""
```

OLD SEPARATE KPF STEPS:
```python
@CompileTF
def get_kpf_weights(self, best_idx):
    rt_dim1, rt_dim2, rt_dim3 = self.rollout_trajectories.shape

    x = self.rollout_trajectories[:, :, 5]
    y = self.rollout_trajectories[:, :, 6]

    d_x = x[:, None] - x
    d_y = y[:, None] - y

    distances = tf.sqrt(tf.square(d_x) + tf.square(d_y))

    distances_sum = tf.reduce_sum(distances, axis=2)

    # distances_sum = distances[:, :, 0]

    # g_distances = 1 - tf.exp(-distances_sum ** 2 / (2 * self.kpf_g_sigma ** 2))
    g_distances = distances_sum

    g_distances = tf.linalg.set_diag(g_distances, tf.ones(rt_dim1) * np.inf)
    divergence_metric = tf.reduce_min(g_distances, axis=1)

    """g_distances = tf.linalg.set_diag(g_distances, tf.zeros(rt_dim1))
    divergence_metric = tf.reduce_mean(g_distances, axis=1)"""

    # METHOD 1 - trajectory similarity using kernels-------------------------------------------------------
    # becomes (n_rollouts x n_chosen_output_states)
    """squeezed_rt = tf.reshape(self.rollout_trajectories[:, :, 5:7], (rt_dim1, rt_dim2 * 2))
    distances = tf.norm(squeezed_rt[:, None] - squeezed_rt, axis=-1)

    # width of Gaussian kernel and distances
    g_distances = 1 - tf.exp(-distances ** 2 / (2 * self.kpf_g_sigma ** 2))
    g_distances = tf.linalg.set_diag(g_distances, tf.ones(rt_dim1) * np.inf)  # np.inf if not reduce_min below!

    # find the closest similarity to any neighbor, use that as a divergence metric
    divergence_metric = tf.reduce_min(g_distances, axis=1)
    # divergence_metric = tf.reduce_mean(g_distances, axis=1)"""
    # -------------------------------------------------------------------------------------------------------

    # METHOD 2 - calculate the distances between endpoints--------------------------------------------------
    """reshaped_rt = tf.reshape(self.rollout_trajectories[:, rt_dim2 - 1, 5:7], (rt_dim1, 1, 2))
    end_rollout_trajectories = tf.squeeze(reshaped_rt, axis=1)
    distances = tf.norm(end_rollout_trajectories[:, None] - end_rollout_trajectories, axis=-1)

    distances = tf.linalg.set_diag(distances, tf.ones(rt_dim1) * np.inf)
    divergence_metric = tf.reduce_min(distances, axis=1)"""

    # get threshold distance for resampling
    # threshold_distance = tf.cast(tf.reduce_max(worst_values), dtype=tf.float32)
    # -------------------------------------------------------------------------------------------------------

    # find indices of furthest (best) points according to predefined kpf_keep_number
    # divergence_metric decreases with more similarity
    sorted_indices = tf.argsort(divergence_metric)
    furthest_indices = sorted_indices[-self.kpf_keep_number:]
    # furthest_indices = tf.math.top_k(divergence_metric, k=self.kpf_keep_number).indices
    total_keep_idx = tf.concat([furthest_indices, best_idx[:self.kpf_keep_best]], 0)
    total_keep_idx, _ = tf.unique(total_keep_idx)
    total_keep_idx = tf.cast(total_keep_idx, tf.int32)

    num_resample = self.num_rollouts - len(total_keep_idx)

    # ALTERNATIVE 1: simple resampling from given distribution--------------------------------------
    # Qres = self.sample_actions(self.rng, self.num_rollouts - len(total_keep_idx))
    # -----------------------------------------------------------------------------------------------

    # ALTERNATIVE 2: smart KPF resampling using weights--------------------------------------------------
    # update weights and normalize
    kpf_weights = divergence_metric / tf.reduce_sum(divergence_metric)

    return kpf_weights, num_resample, total_keep_idx, sorted_indices

@CompileTF
def get_kpf_samples(self, Q, noise, sorted_indices):
    # return self.sample_actions(self.rng, self.kpf_num_resample)

    # return self.sample_actions(self.rng, self.kpf_num_resample)
    perturb_indices = sorted_indices[-self.kpf_perturb_best:]

    ratio = tf.cast(tf.math.ceil(self.kpf_num_resample / self.kpf_perturb_best), dtype=tf.int32)
    tiled_pi = tf.tile(perturb_indices, [ratio])[:self.kpf_num_resample]

    Q_picked = tf.gather(Q, tiled_pi)
    Q_final = tf.add(Q_picked, noise)
    return Q_final
```