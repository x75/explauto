from numpy import array, pi, sqrt, cos, sin, linspace, zeros
from numpy import random
import numpy as np


from ...exceptions import ExplautoNoTestCasesError
from .pointmass import PointmassEnvironment


def make_pointmass_config(st_ndims, m_ndims, s_ndims, sensor_transform,
                          m_max, s_mins, s_maxs, mass, sysnoise, sensnoise,
                          doRos = False, control_order = 2, friction = 0.01, motor_aberration = {}):
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
                sensnoise = sensnoise,
                doRos = doRos,
                control_order = control_order,
                friction = friction,
                motor_aberration = motor_aberration
    )

low_dim_vel = make_pointmass_config(st_ndims = 3,
                                m_ndims = 1,
                                s_ndims = 1,
                                sensor_transform = np.array([[0, 1, 0]]),
                                m_max = 1.0,
                                s_mins = array([-1.0]),
                                s_maxs = array([ 1.0]),
                                mass = 1,
                                sysnoise = 0.00,
                                sensnoise = 0.00,
                                # motor_aberration = {"type": "sin", "coef": 0.4, "noise": True, "noise_mu": 0.0, "noise_sigma": 0.01}
                                # motor_aberration = {"type": "tanh", "coef": 10.0, "noise": True, "noise_mu": 0.0, "noise_sigma": 0.01}
                                # motor_aberration = {"type": "exp", "coef": 0.6, "noise": True, "noise_mu": 0.0, "noise_sigma": 0.01}
                                # motor_aberration = {"noise": True, "noise_mu": 0.0, "noise_sigma": 0.01}
                                )

low_dim_full = make_pointmass_config(st_ndims = 3,
                                m_ndims = 1,
                                s_ndims = 2, # pos, vel
                                sensor_transform = np.array([[1, 0, 0], [0, 1, 0]]), # np.eye(3),
                                m_max = 1,
                                s_mins = array([-1.0, -1.0]),
                                s_maxs = array([ 1.0,  1.0]),
                                mass = 1,
                                sysnoise = 0.02,
                                sensnoise = 0.02)

# mid dimensional
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
                                mass = 1.0,
                                sysnoise = 0.005,
                                sensnoise = 0.005,
                                doRos = False)

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

high_dim_vel = make_pointmass_config(st_ndims = 3 * 10,
                                m_ndims = 10,
                                s_ndims = 10,
                                sensor_transform = np.array([
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    ]),
                                m_max = 1,
                                s_mins = array([-1.0] *  10),
                                s_maxs = array([ 1.0] *  10),
                                mass = 1.0,
                                sysnoise = 0.0,
                                sensnoise = 0.0)

# smq, implies _c2 control 2nd order (force) with respect to position
low_dim_acc_vel = make_pointmass_config(st_ndims = 3,
                                m_ndims = 1,
                                s_ndims = 2,
                                sensor_transform = np.array([
                                    [0, 1, 0],
                                    [0, 0, 1]
                                    ]),
                                m_max = 1.0,
                                s_mins = array([-1.0]),
                                s_maxs = array([ 1.0]),
                                mass = 1,
                                sysnoise = 0.00,
                                sensnoise = 0.00)

# smq, this is _c1, control 1st order (velocity) with respect to position
low_dim_acc_vel_c1 = make_pointmass_config(st_ndims = 3,
                                m_ndims = 1,
                                s_ndims = 2,
                                sensor_transform = np.array([
                                    [0, 1, 0],
                                    [0, 0, 1]
                                    ]),
                                m_max = 1.0,
                                s_mins = array([-1.0]),
                                s_maxs = array([ 1.0]),
                                mass = 1,
                                sysnoise = 0.00,
                                sensnoise = 0.00,
                                control_order = 1)

# smq
planar_dim_acc_vel = make_pointmass_config(st_ndims = 6,
                                m_ndims = 2,
                                s_ndims = 4,
                                sensor_transform = np.array([
                                    [0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1],
                                ]),
                                m_max = 1.0,
                                s_mins = array([-1.0, -1.0]),
                                s_maxs = array([ 1.0,  1.0]),
                                mass = 1,
                                sysnoise = 0.00,
                                sensnoise = 0.00)
# smq
mid_dim_acc_vel = make_pointmass_config(st_ndims = 9,
                                m_ndims = 3,
                                s_ndims = 6,
                                sensor_transform = np.array([
                                    # [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    # [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    # [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 1],
                                    ]),
                                m_max = 1.0,
                                s_mins = array([-1.0] *  3),
                                s_maxs = array([ 1.0] *  3),
                                mass = 1.0,
                                sysnoise = 0.0,
                                sensnoise = 0.0,
                                doRos = False,
                                friction = 0.01,
                                )
# smq
high_dim_acc_vel = make_pointmass_config(st_ndims = 3 * 10,
                                m_ndims = 10,
                                s_ndims = 2 * 10,
                                sensor_transform = np.array([
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                    ]),
                                m_max = 1,
                                s_mins = array([-1.0] *  10),
                                s_maxs = array([ 1.0] *  10),
                                mass = 1.0,
                                sysnoise = 0.0,
                                sensnoise = 0.0)

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
    'low_dim_vel':  low_dim_vel,
    'mid_dim_vel':  mid_dim_vel,
    'high_dim_vel': high_dim_vel,
    # smq
    'low_dim_acc_vel':     low_dim_acc_vel,
    'planar_dim_acc_vel':  planar_dim_acc_vel,
    'mid_dim_acc_vel':     mid_dim_acc_vel,
    'high_dim_acc_vel':     high_dim_acc_vel,
    # smq kinematic
    'low_dim_acc_vel_c1':     low_dim_acc_vel_c1,
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
