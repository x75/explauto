from numpy import array
from numpy import random
from copy import copy

import pointmasslib

from ..environment import Environment

class PointmassEnvironment(Environment):
    def __init__(self, m_mins, m_maxs, s_mins, s_maxs, noise, force_max = 1, mass = 1.0, dt = 0.1):
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)

        self.noise = noise
        self.x0 = [0, 0]
        self.dt = dt
        self.force_max = force_max
        self.x = copy(self.x0)

    def compute_motor_command(self, ag_state):
        return bounds_min_max(ag_state, self.conf.m_mins, self.conf.m_maxs)

    def reset(self):
        self.x = copy(self.x0)

    def apply_force(self, u):
        self.x = pointmasslib.simulate(self.x, [u], self.dt)

    def compute_sensori_effect(self, m):
        # for u in self
        self.apply_force(m)
        res = array(self.x)
        print res
        print self.noise
        # print *res.shape
        res += self.noise * random.randn(*res.shape)
        return res

    def plot_current_state(self, ax):
        # ax.plot(
        print("%s.plot_current_state: implement me" % (self.__class__.__name__))
