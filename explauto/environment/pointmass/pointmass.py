from numpy import array
from numpy import random
import numpy as np
from copy import copy

import pointmasslib

from ..environment import Environment

class PointmassEnvironment(Environment):
    def __init__(self, st_ndims, m_ndims, s_ndims, sensor_transform, m_mins, m_maxs, s_mins, s_maxs, sysnoise = 0.0, sensnoise = 0.0, force_max = 1, mass = 1.0, dt = 0.1):
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)

        # FIXME: dimensions
        # - world/state dimension
        self.state_dim = st_ndims # pos, vel, acc
        self.world_dim = st_ndims / 3
        # - motor dimension
        self.motor_dim = m_ndims
        # - sensor/observable dimension
        self.sensor_dim = s_ndims
        self.sensor_transform = sensor_transform # state to sensor transformation matrix
                
        print "self.conf", self.conf
        self.sysnoise  = sysnoise
        self.sensnoise = sensnoise
        # self.spacedims = self.conf.s_ndims/2
        self.x0 = np.zeros((self.state_dim, 1)) # [0, 0] * self.conf.ndims
        self.dt = dt
        self.force_max = force_max
        self.mass = mass
        self.x = self.x0.copy()

    def compute_motor_command(self, ag_state):
        return bounds_min_max(ag_state, self.conf.m_mins, self.conf.m_maxs)

    def reset(self):
        self.x = self.x0.copy()

    def apply_force(self, u):
        # print "u", u
        # FIXME: insert motor transfer function
        a = (u/self.mass).reshape((self.world_dim, 1))
        # print("a.shape", a.shape)
        # print "a", a, self.x[self.conf.s_ndims/2:]
        v = self.x[self.world_dim:self.world_dim*2] + a * self.dt
        # print "v", v
        p = self.x[:self.world_dim] + v * self.dt

        self.x[:self.world_dim] = p.copy()
        self.x[self.world_dim:self.world_dim*2] = v.copy()
        self.x[self.world_dim*2:] = a.copy()

        self.x += self.sysnoise * random.randn(self.x.shape[0], self.x.shape[1])
        
        # return x
        # self.x = x # pointmasslib.simulate(self.x, [u], self.dt)

    def step(self, m):
        return self.compute_sensori_effect(m)

    def compute_sensori_effect(self, m):
        # for u in self
        self.apply_force(m)
        # res = array(self.x)
        # print res
        # print self.noise
        # print "self.x.shape", self.x.shape
        sensors = np.dot(self.sensor_transform, self.x)
        sensors += self.sensnoise * random.normal(0, 1.0, (self.sensor_dim, 1))
        return sensors

    def plot_current_state(self, ax):
        # ax.plot(
        print("%s.plot_current_state: implement me" % (self.__class__.__name__))
