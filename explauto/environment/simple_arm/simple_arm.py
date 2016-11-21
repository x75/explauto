import numpy as np

from ..environment import Environment
from ...utils import bounds_min_max


def forward(angles, lengths):
    """ Link object as defined by the standard DH representation.

    :param list angles: angles of each joint

    :param list lengths: length of each segment

    :returns: a tuple (x, y) of the end-effector position

    .. warning:: angles and lengths should be the same size.
    """
    x, y = joint_positions(angles, lengths)
    return x[-1], y[-1]


def joint_positions(angles, lengths, unit='rad'):
    """ Link object as defined by the standard DH representation.

    :param list angles: angles of each joint

    :param list lengths: length of each segment

    :returns: x positions of each joint, y positions of each joints, except the first one wich is fixed at (0, 0)

    .. warning:: angles and lengths should be the same size.
    """
    if len(angles) != len(lengths):
        raise ValueError('angles and lengths must be the same size!')

    if unit == 'rad':
        a = np.array(angles)
    elif unit == 'std':
        a = np.pi * np.array(angles)
    else:
        raise NotImplementedError
     
    a = np.cumsum(a)
    return np.cumsum(np.cos(a)*lengths), np.cumsum(np.sin(a)*lengths)


def lengths(n_dofs, ratio):
    l = np.ones(n_dofs)
    for i in range(1, n_dofs):
        l[i] = l[i-1] / ratio
    return l / sum(l)


class SimpleArmEnvironment(Environment):
    use_process = True

    def __init__(self, m_mins, m_maxs, s_mins, s_maxs,
                 length_ratio, noise):
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)

        self.length_ratio = length_ratio
        self.noise = noise

        self.factor = 1.0

        self.lengths = lengths(self.conf.m_ndims, self.length_ratio)

    def compute_motor_command(self, joint_pos_ag):
        # print "joint_pos_ag", joint_pos_ag
        joint_pos_ag *= self.factor
        return bounds_min_max(joint_pos_ag, self.conf.m_mins, self.conf.m_maxs)

    def compute_sensori_effect(self, joint_pos_env):
        hand_pos = np.array(forward(joint_pos_env, self.lengths))
        hand_pos += self.noise * np.random.randn(*hand_pos.shape)
        return hand_pos

    def plot(self, ax, m, s, **kwargs_plot):
        self.plot_arm(ax, m, **kwargs_plot)

    def plot_arm(self, ax, m, **kwargs_plot):
        x, y = joint_positions(m, self.lengths)
        x, y = [np.hstack((0., a)) for a in x, y]
        ax.plot(x, y, 'grey', lw=2, **kwargs_plot)
        ax.plot(x[0], y[0], 'ok', ms=6)
        ax.plot(x[-1], y[-1], 'sk', ms=6)
        ax.axis([self.conf.s_mins[0], self.conf.s_maxs[0], self.conf.s_mins[1], self.conf.s_maxs[1]])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

# from explauto.environment.simple_arm import SimpleArmEnvironment

class SimpleDynamicArmEnvironment(SimpleArmEnvironment):
    def __init__(self, **kwargs):
        SimpleArmEnvironment.__init__(self, **kwargs)
        # self.current_context = [0] # 0: lazy, 1: normal
        self.current_context = [0 for n in range(self.conf.s_ndims)]
        self.m = np.zeros((self.conf.m_ndims,)) # [0 for n in range(self.conf.m_ndims)]
        self.st = [0 for n in range(self.conf.s_ndims)]
        self.ds = [0 for n in range(self.conf.s_ndims)]

        if True:
            import rospy
            from std_msgs.msg import Float32MultiArray
            rospy.init_node("simplearm_environment")
            self.pubs = {}
            self.pubs["pos"] = rospy.Publisher("/robot/0/pos", Float32MultiArray)
            self.msgs = {}
            self.msgs["pos"] = Float32MultiArray()
        
        
    def compute_motor_command(self, m):
        # print("SimpleDynamicArmEnvironment: compute_motor_command", type(m), type(self.m))
        # if self.current_context[0]:
        #     return m
        # else:
        #     return [mi/4 for mi in m]
        self.m += m
        return self.m 
        
    def compute_sensori_effect(self, m):
        # self.m += m
        s = SimpleArmEnvironment.compute_sensori_effect(self, m)
        
        if True:
            self.msgs["pos"].data = []
            # print self.x.tolist()
            self.msgs["pos"].data = (s * 1.0).flatten().tolist()
            self.msgs["pos"].data.append(0)
            self.pubs["pos"].publish(self.msgs["pos"])
        # self.s -= s
        # self.ds = self.s - s
        # self.s = s
        # print("s", s)
        # self.current_context[0] = 1 - self.current_context[0]
        self.current_context = s # self.ds
        return s
    
