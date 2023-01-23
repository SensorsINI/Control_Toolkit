from SI_Toolkit.computation_library import ComputationLibrary, NumpyLibrary, TensorFlowLibrary, PyTorchLibrary

import numpy as np
import timeit


class Interpolator:

    def __init__(self, horizon: int,
                 period_interpolation_inducing_points: int,
                 num_control_inputs: int,
                 computational_library: ComputationLibrary,
                 algorithm='Diego',
                 ):

        if horizon < period_interpolation_inducing_points:
            print(
                "Chosen horizon is smaller than period for interpolation points. It will give effect of clamped range of interpolator output")
        self.horizon = horizon
        self.period_interpolation_inducing_points = period_interpolation_inducing_points
        self.lib = computational_library
        self.num_control_inputs = num_control_inputs

        self.number_of_interpolation_inducing_points = \
            self.get_number_of_interpolation_inducing_points(self.horizon, self.period_interpolation_inducing_points)

        self.last_point_index = (self.number_of_interpolation_inducing_points-1) * self.period_interpolation_inducing_points  # Assuming first is 0

        self.evaluate_at = self.lib.cast(self.lib.arange(self.horizon), self.lib.float32)

        self.interp_mat = None
        if self.period_interpolation_inducing_points == 1:
            self.interpolate = self.no_interpolation
        if algorithm == 'Diego':
            self.calculate_interpolation_matrix()
            self.interpolate = self._interpolate_Diego
        else:
            if self.lib == NumpyLibrary:
                from scipy import interpolate
                self.interpolate_f = interpolate.interp1d
                self.interpolate = self._interpolate_np
            if self.lib == TensorFlowLibrary:
                import tensorflow_probability as tfp
                self.interpolate_f = tfp.math.interp_regular_1d_grid
                self.interpolate = self._interpolate_tf
            if self.lib == PyTorchLibrary:
                raise NotImplementedError('There seems to be no generic interpolation function in Pytorch yet')

    @staticmethod
    def no_interpolation(y):
        return y

    def calculate_interpolation_matrix(self):
        step = self.period_interpolation_inducing_points
        self.interp_mat = np.zeros(
            (
                (self.number_of_interpolation_inducing_points - 1) * step + 1,
                self.number_of_interpolation_inducing_points,
                self.num_control_inputs,
            ),
            dtype=np.float32,
        )
        step_block = np.zeros((step, 2, self.num_control_inputs), dtype=np.float32)
        for j in range(step):
            step_block[j, 0, :] = (step - j) * np.ones(
                self.num_control_inputs, dtype=np.float32
            )
            step_block[j, 1, :] = j * np.ones(
                self.num_control_inputs, dtype=np.float32
            )
        for i in range(self.number_of_interpolation_inducing_points - 1):
            self.interp_mat[i * step: (i + 1) * step, i: i + 2, :] = step_block
        self.interp_mat[-1, -1, :] = 1
        self.interp_mat = self.interp_mat[: self.horizon, :, :] / step
        self.interp_mat = self.lib.constant(
            self.lib.permute(self.lib.to_tensor(self.interp_mat, self.lib.float32), (1, 0, 2)), self.lib.float32
        )

    def get_number_of_interpolation_inducing_points(self, horizon, period_interpolation_inducing_points):
        # The two equations are  equivalent, you can use whichever
        number_of_interpolation_inducing_points = self.lib.ceil(
            (horizon - 1) / period_interpolation_inducing_points) + 1
        # number_of_interpolation_inducing_points = self.lib.floor((horizon-2)/period_interpolation_inducing_points) + 2
        return int(number_of_interpolation_inducing_points)

    def _interpolate_tf(self, y, axis=1):

        y_interp = self.interpolate_f(self.evaluate_at, 0.0, self.last_point_index, y, axis=axis)
        return y_interp

    def _interpolate_np(self, y, axis=1):
        f = self.interpolate_f(
            np.arange(0.0, self.last_point_index+1, step=self.period_interpolation_inducing_points), y, axis=axis)
        y_interp = f(self.evaluate_at)
        return y_interp

    def _interpolate_Diego(self, y):

        y_interp = self.lib.permute(
            self.lib.matmul(
                self.lib.permute(y, (2, 0, 1)),
                self.lib.permute(self.interp_mat, (2, 0, 1)),
            ),
            (1, 2, 0),
        )
        return y_interp


if __name__ == '__main__':

    HORIZON = 43
    PERIOD_INTERPOLATION_INDUCING_POINTS = 10
    COMPUTATION_LIBRARIES = [NumpyLibrary, TensorFlowLibrary, PyTorchLibrary]
    NUM_CONTROL_INPUTS = 2
    ALGORITHMS = ['Diego', 'Standard']

    BATCH  = 2000

    for COMPUTATION_LIBRARY in COMPUTATION_LIBRARIES:
        for ALGORITHM in ALGORITHMS:

            print('**********************************************************************')
            print(COMPUTATION_LIBRARY.lib)
            print(ALGORITHM)
            LIB = COMPUTATION_LIBRARY
            I: Interpolator = Interpolator(HORIZON,
                                           PERIOD_INTERPOLATION_INDUCING_POINTS,
                                           NUM_CONTROL_INPUTS,
                                           COMPUTATION_LIBRARY,
                                           ALGORITHM)

            points = I.number_of_interpolation_inducing_points

            Y = np.random.normal(size= [BATCH, points, NUM_CONTROL_INPUTS])

            Y = LIB.to_tensor(Y, LIB.float32)
            f = I.interpolate

            setup = 'from __main__ import f, Y \nY_interp = f(Y)'
            print(timeit.timeit('Y_interp = f(Y)', setup, number=10 ** 4))

            pass


