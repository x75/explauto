from numpy import array, pi, eye

from ...exceptions import ExplautoNoTestCasesError
from .morse_copter import CopterMorseEnvironment

def make_morse_conf(m_ndims, s_ndims, sensor_transform, m_mins, m_maxs, s_mins, s_maxs):
    return dict(
        m_ndims = m_ndims,
        s_ndims = s_ndims,
        sensor_transform = sensor_transform,
        m_mins = m_mins, # array([-m_max] * m_ndims),
        m_maxs = m_maxs, # array([m_max] * m_ndims),
        s_mins = s_mins,
        s_maxs = s_maxs,
    )

copter_attitude = make_morse_conf(
    m_ndims = 4, # roll, pitch, yaw, thrust
    s_ndims = 5, # vx, vy, vz, np.cos(yaw), np.sin(yaw)
    sensor_transform = eye(5),
    m_mins = array([-pi/16, -pi/16, -pi/16, 0.3]),
    m_maxs = array([ pi/16,  pi/16,  pi/16, 0.7]),
    s_mins = array([-1, -1, -1, -1, -1]),
    s_maxs = array([ 1,  1,  1,  1,  1])
    )

environment = CopterMorseEnvironment
configurations = {
    "copter_attitude": copter_attitude,
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
