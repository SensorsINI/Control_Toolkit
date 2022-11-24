
class TargetPositionGenerator:
    def __init__(self, lib, horizon):
        self.horizon = horizon
        self.lib = lib

    def step(self, time):
        return self.lib.zeros(self.horizon)+self.lib.sin(time/1.3)
