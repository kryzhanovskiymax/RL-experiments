import gym
import numpy as np
from dataclasses import dataclass

@dataclass
class CannonEnvSettings():
    angle: float = 45.0
    degrees: bool = True
    distance: float = 100.0
    g: float = 10


class CannonEnv(gym.Env):
    def __init__(self,
                 settings: CannonEnvSettings = CannonEnvSettings()):
        super(CannonEnv, self).__init__()
        if settings.degrees:
            self.alpha = np.deg2rad(settings.angle)
        else:
            self.alpha = settings.angle
        self.distance = settings.distance
        self.g = settings.g
        self.action_space = gym.spaces.Box(low=0,
                                       high=np.inf,
                                       shape=(1,),
                                       dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0,
                                            high=2*np.pi,
                                            shape=(1,),
                                            dtype=np.float32)

    def reset(self):
        return self.alpha, {}

    def _calc_delta(self, speed):
        s = 2 * (speed ** 2) * np.sin(self.alpha) * np.cos(self.alpha) / self.g
        delta = self.distance - s
        return delta

    def step(self, action):
        delta = self._calc_delta(action)
        return self.alpha, -np.abs(delta), True, {'delta': delta}


gym.envs.registration.register(
    id='CannonEnv-v0',
    entry_point=CannonEnv
)