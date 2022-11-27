# stub for genrating desired future trajectory of cartpole

class TrajectoryGenerator:
    def __init__(self, lib, horizon:int):
        """ Construct the trajectory generator.

        :param lib: the computation library, e.g. tensorflow
        :param horizon: the MPC horizon in timesteps
        """
        self.horizon = horizon
        self.lib = lib

    def step(self, time):
        """ Computes the desired future state trajectory at this time.

        :param time: the scalar time in seconds

        :returns: the target state trajectory of cartpole.
        It should be a Tensor with NaN entries for don't care states, and otherwise the desired state values.

        """
        return self.lib.zeros(self.horizon)+self.lib.sin(time/1.3)
