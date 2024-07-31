import abc
import random
import gym
import numpy as np
import warnings

from itertools import chain
from enum import Enum
from gym import spaces
from gym.utils import seeding


class Player(Enum):
    X = 1
    O = 2

class TicTacToeEnv(gym.Env)