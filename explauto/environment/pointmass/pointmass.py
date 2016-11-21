from numpy import array
from numpy import random
import numpy as np
from copy import copy

# import pointmasslib

from ..environment import Environment

from ...utils import bounds_min_max

class PointmassEnvironment(Environment):
    def __init__(self, st_ndims, m_ndims, s_ndims, sensor_transform, m_mins, m_maxs, s_mins, s_maxs, sysnoise = 0.0, sensnoise = 0.0, force_max = 1, mass = 1.0, dt = 0.1, doRos = False, control_order = 2, friction = 0.01):
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)

        # FIXME: dimensions
        # - world/state dimension
        self.state_dim = st_ndims # pos, vel, acc
        self.world_dim = st_ndims / 3
        # - motor dimension
        self.m_ndims = m_ndims
        # - sensor/observable dimension
        self.s_ndims = s_ndims
        self.sensor_transform = sensor_transform # state to sensor transformation matrix

        # check wether configured number of sensors matches sensor transform
        assert(self.sensor_transform.shape[0] == self.s_ndims)
                
        # print "%s.__init__: self.conf = %s" % (self.__class__.__name__, self.conf)
        self.sysnoise  = sysnoise
        self.sensnoise = sensnoise
        # self.spacedims = self.conf.s_ndims/2
        self.x0 = np.zeros((self.state_dim, 1)) # [0, 0] * self.conf.ndims
        self.x_ = np.zeros((self.world_dim,  )) # [0, 0] * self.conf.ndims
        self.dt = dt
        self.force_max = force_max
        self.mass = mass
        self.friction = friction
        self.x = self.x0.copy()
        # control mode: kinematic, dynamic
        self.control_order = control_order

        self.sm_delays = dict()
        for i in range(self.s_ndims):
            self.sm_delays[i] = 0
        self.a_ = np.zeros((self.world_dim, 1))

        # context
        self.current_context = np.dot(self.sensor_transform, self.x).flatten()
        self.cnt = 0

        self.doRos = doRos

        self.pubfuncs = []
                
        if self.doRos:
            import rospy
            from std_msgs.msg import Float32MultiArray
            rospy.init_node("pointmass_environment")
            self.pubs = {}
            self.pubs["pos"] = rospy.Publisher("/robot/0/pos", Float32MultiArray)
            self.msgs = {}
            self.msgs["pos"] = Float32MultiArray()
            self.pubfuncs.append(self.pub_ros)
        
    def compute_motor_command(self, ag_state):
        # print "%s.compute_motor_command arg ag_state %s" % (self.__class__.__name__, ag_state)
        return bounds_min_max(ag_state, self.conf.m_mins, self.conf.m_maxs)

    def reset(self):
        self.x = self.x0.copy()

    def pub_ros(self):
        if self.doRos:
            self.msgs["pos"].data = []
            # print self.x.tolist()
            # self.msgs["pos"].data = (self.x * 0.1).flatten().tolist()
            self.msgs["pos"].data = self.x_.flatten().tolist()
            self.pubs["pos"].publish(self.msgs["pos"])

    def pub(self):
        for p in self.pubfuncs:
            p()

    def scale(self):
        # FIXME: what is that?
        for i in range(self.world_dim):
            self.x_[i] = self.x[i,0] * 0.1
            # self.x_[1] = self.x[1,0] * 0.1
            # self.x_[2] = self.x[2,0] * 0.1
            
    def apply_force(self, u):
        """control pointmass with force command (2nd order)"""
        # print "u", u
        # FIXME: insert motor transfer function
        a = (u/self.mass).reshape((self.world_dim, 1))
        # a += np.random.normal(0.05, 0.01, a.shape)

        # # world modification
        # if np.any(self.x[:self.world_dim] > 0):
        #     a += np.random.normal(0.05, 0.01, a.shape)
        # else:
        #     a += np.random.normal(-0.1, 0.01, a.shape)
            
        # print("a.shape", a.shape)
        # print "a", a, self.x[self.conf.s_ndims/2:]
        v = self.x[self.world_dim:self.world_dim*2] * (1 - self.friction) + a * self.dt
        
        # self.a_ = a.copy()
        
        
        # # world modification
        # v += np.sin(self.cnt * 0.01) * 0.05
        
        # print "v", v
        p = self.x[:self.world_dim] + v * self.dt

        self.x[:self.world_dim] = p.copy()
        self.x[self.world_dim:self.world_dim*2] = v.copy()
        self.x[self.world_dim*2:] = a.copy()

        self.x += self.sysnoise * random.randn(self.x.shape[0], self.x.shape[1])

        # print "self.x[2,0]", self.x[2,0]

        self.scale()
        self.pub()                
        self.cnt += 1
        
        # return x
        # self.x = x # pointmasslib.simulate(self.x, [u], self.dt)

    def apply_vel(self, u):
        """control pointmass with velocity command (1st order)"""
        v = u # self.x[self.world_dim:self.world_dim*2] * (1 - self.friction) + a * self.dt
        p = self.x[:self.world_dim] + v * self.dt
        a = (v - self.x[self.world_dim:self.world_dim*2]) / self.dt

        self.x[:self.world_dim] = p.copy()
        self.x[self.world_dim:self.world_dim*2] = v.copy()
        self.x[self.world_dim*2:] = a.copy()

        self.x += self.sysnoise * random.randn(self.x.shape[0], self.x.shape[1])
        
        self.scale()
        self.pub()                
        self.cnt += 1
        
    def step(self, m):
        return self.compute_sensori_effect(m)

    def compute_sensori_effect(self, m):
        if self.control_order == 2:
            self.apply_force(m)
        elif self.control_order == 1:
            self.apply_vel(m)
        elif self.control_order == 0:
            self.apply_pos(m)
            
        # create measurements as linear combinations from states via sensor_transform matrix
        sensors = np.dot(self.sensor_transform, self.x) # .flatten()
        sensors += self.sensnoise * random.normal(0, 1.0, sensors.shape) # (self.sensor_dim, 1)
        sensors = sensors.flatten()
        self.current_context = sensors.copy()
        # print("sensors.shape", sensors.shape)
        return sensors # .tolist()

    def plot_current_state(self, ax):
        # ax.plot(
        print("%s.plot_current_state: implement me" % (self.__class__.__name__))
