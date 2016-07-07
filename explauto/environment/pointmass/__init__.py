from numpy import array, pi, sqrt, cos, sin, linspace, zeros
from numpy import random
import numpy as np


from ...exceptions import ExplautoNoTestCasesError
from .pointmass import PointmassEnvironment


def make_pointmass_config(st_ndims, m_ndims, s_ndims, sensor_transform, m_max, s_mins, s_maxs, mass, sysnoise, sensnoise):
    return dict(st_ndims = st_ndims,
                m_ndims = m_ndims,
                s_ndims = s_ndims,
                sensor_transform = sensor_transform,
                m_mins=array([-m_max] * m_ndims),
                m_maxs=array([m_max] * m_ndims),
                s_mins=s_mins,
                s_maxs=s_maxs,
                mass=mass,
                sysnoise = sysnoise,
                sensnoise = sensnoise
    )

low_dim_vel = make_pointmass_config(st_ndims = 3,
                                m_ndims = 1,
                                s_ndims = 1,
                                sensor_transform = np.array([[0, 1, 0]]),
                                m_max = 1.0,
                                s_mins = array([-1.0]),
                                s_maxs = array([ 1.0]),
                                mass = 1,
                                sysnoise = 0.0,
                                sensnoise = 0.0)

low_dim_full = make_pointmass_config(st_ndims = 3,
                                m_ndims = 1,
                                s_ndims = 3,
                                sensor_transform = np.eye(3),
                                m_max = 1,
                                s_mins = array([-1.0, -1.0, -1.0] * 3),
                                s_maxs = array([ 1.0,  1.0,  1.0] * 3),
                                mass = 1,
                                sysnoise = 0.02,
                                sensnoise = 0.02)

mid_dim_vel = make_pointmass_config(st_ndims = 9,
                                m_ndims = 3,
                                s_ndims = 3,
                                sensor_transform = np.array([
                                    [0, 0, 0, 1, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 1, 0, 0, 0]]),
                                m_max = 1,
                                s_mins = array([-1.0] *  3),
                                s_maxs = array([ 1.0] *  3),
                                mass = 1,
                                sysnoise = 0.02,
                                sensnoise = 0.02)

mid_dim_full = make_pointmass_config(st_ndims = 9,
                                m_ndims = 3,
                                s_ndims = 9,
                                sensor_transform = np.eye(9),
                                m_max = 1,
                                s_mins = array([-10.0, -10.0, -10.0] *  3),
                                s_maxs = array([ 10.0,  10.0,  10.0] *  3),
                                mass = 1,
                                sysnoise = 0.02,
                                sensnoise = 0.02)

# hd_dim = make_pointmass_config(st_ndims = 30,
#                                m_ndims = 10,
#                                s_ndims = 10,
#                                sensor_transform = np.array([
#                                    [0] * 10 + [1] + [0] * 19,
#                                ]),
#                                 1, array([-10.0, -10.0] * 10), array([10., 10.] * 10), 1, 0.02, 0.02)
# hd_dim = make_arm_config(30, pi/8., array([-0.6, -0.9]), array([1., 0.9]), 1., 0.001)

# hd_dim_range = make_arm_config(30, pi/8., array([-2., -2.]), array([2., 2.]), 1., 0.001)
# hd_dim_range = make_arm_config(15, pi/3., array([-2., -2.]), array([2., 2.]), 2./3., 0.001)
# hd_dim_range = make_arm_config(50, pi/12., array([-2., -2.]), array([2., 2.]), 1., 0.02)

environment = PointmassEnvironment
configurations = {
    'low_dim_full': low_dim_full,
    'mid_dim_full': mid_dim_full,
    'low_dim_vel': low_dim_vel,
    'mid_dim_vel': mid_dim_vel,
    # 'high_dimensional': hd_dim,
    # 'high_dim_high_s_range': hd_dim_range,
    'default': low_dim_vel
}


def testcases(config_str, n_samples=100):
    tests = zeros((n_samples, 2))
    # #FIXME low_dimensional
    # if config_str in ('high_dimensional', 'high_dim_high_s_range'):
    #     i = 0
    #     for r, theta in array([1., 2*pi]) * random.rand(n_samples, 2) + array([0., -pi]):
    #         tests[i, :] = sqrt(r) * array([cos(theta), sin(theta)])
    #         i += 1
    #     return tests

    # else:
    env = environment(**configurations[config_str])
    env.sysnoise = 0.
    env.sensnoise = 0.
    return env.uniform_sensor(n_samples)
