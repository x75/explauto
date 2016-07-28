"""explauto Environment for a robot (quadrotor) simulated with MORSE"""

import numpy as np

import subprocess, time

from ..environment import Environment

from ...utils import bounds_min_max

import rospy
# from geometry_msgs.msg import Twist, Wrench, Point, Pose, Quaternion, Vector3
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray
from tf.transformations import euler_from_quaternion, quaternion_from_euler

def reset_simulation():
    """reset MORSE simulation by teleporting robot back to initial position"""
    # call a pymorse (py3 only) script externally to do the teleporting and simulation control
    py3env = {"PYTHONPATH": "/usr/local/lib/python3/dist-packages"}
    ret = subprocess.Popen(["python3",  "../im/im_quadrotor_controller_reset_pymorse.py"], env = py3env)
    # wait for reset to happen
    time.sleep(0.5)
    return

class CopterMorseEnvironment(Environment):
    def __init__(self, m_ndims, s_ndims, m_mins, m_maxs, s_mins, s_maxs, sensor_transform):
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)
        rospy.init_node("coptermorseenvironment")
        # assert motor dim == control mode
    
        self.m_ndims = m_ndims
        self.s_ndims = s_ndims
        self.sensor_transform = sensor_transform
        
        self.x0 = np.zeros((self.s_ndims, ))
        # state / sensors for particular sensorimotor model realization
        self.x = self.x0.copy()
        # helper state for indirect effects ;)
        self.x_ = np.zeros((5, ))
        # copy
        self.current_context = np.dot(self.sensor_transform, self.x).flatten()

        # ros stuff
        self.r = rospy.Rate(10)
        self.pubs = {}
        self.subs = {}

        ctrl_mode = "attctrl"
        ctrl_type = Float32MultiArray
            
        self.attctrl = Float32MultiArray()
        self.attctrl.data = [0 for i in range(self.m_ndims)]

        self.euler = Float32MultiArray()
        self.euler.data = [1, 0]
        
        self.pubs[ctrl_mode] = rospy.Publisher("/quad1/%s" % (ctrl_mode), ctrl_type, queue_size = 2)
        self.pubs["euler"] = rospy.Publisher("/quad1/%s" % ("euler"), Float32MultiArray, queue_size = 2)

        self.subs["odometry"] = rospy.Subscriber("/quad1/odometry", Odometry, self.cb_odometry)
        self.subs["imu"] = rospy.Subscriber("/quad1/imu", Imu, self.cb_imu)

    def cb_odometry(self, msg):
        # print "odom", msg
        # print "msg.twist.twist.linear.z", msg.twist.twist.linear.z
        self.x[0] = msg.twist.twist.linear.x
        self.x[1] = msg.twist.twist.linear.y
        self.x[2] = msg.twist.twist.linear.z
        euler_angles = np.array(euler_from_quaternion([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
            ]))
        # print "euler_angles", euler_angles
        self.x[3] = np.cos(euler_angles[2])
        self.x[4] = np.sin(euler_angles[2])

        self.euler.data[0] = self.x[3]
        self.euler.data[1] = self.x[4]

        self.x_[0] = msg.pose.pose.position.x
        self.x_[1] = msg.pose.pose.position.y
        self.x_[2] = msg.pose.pose.position.z
        
    def cb_imu(self, msg):
        # print "imu", msg
        # self.x[]
        pass
                
    def compute_motor_command(self, ag_state):
        return bounds_min_max(ag_state, self.conf.m_mins, self.conf.m_maxs)

    def reset(self):
        self.x = self.x0.copy()
        reset_simulation()
        
    def compute_sensori_effect(self, m):
        # print("%s.%s: m" % (self.__class__.__name__, "compute_sensori_effect"), m)
        self.attctrl.data = m.tolist()
        self.pubs["attctrl"].publish(self.attctrl)
        self.pubs["euler"].publish(self.euler)
        self.r.sleep()
        # print "self.x", self.x
        sensors = self.x.copy() # np.random.uniform(-1, 1, (self.s_ndims,))
        # print("%s, %s" % (self.__class__.__name__, sensors))
        self.current_context = sensors.copy()
        return sensors

class CopterMorseEnvironmentFull(CopterMorseEnvironment):
    def __init__(self, m_ndims, s_ndims, m_mins, m_maxs, s_mins, s_maxs, sensor_transform):
        CopterMorseEnvironment.__init__(self, m_ndims, s_ndims, m_mins, m_maxs, s_mins, s_maxs, sensor_transform)
        
    def cb_odometry(self, msg):
        assert self.s_ndims == 15 # 3 pos, 3 vel, 3 cos euler, 3 sin euler, 3 ang rates
        # print "odom", msg
        # print "msg.twist.twist.linear.z", msg.twist.twist.linear.z
        self.x[0] = msg.pose.pose.position.x
        self.x[1] = msg.pose.pose.position.y
        self.x[2] = msg.pose.pose.position.z
        self.x[3] = msg.twist.twist.linear.x
        self.x[4] = msg.twist.twist.linear.y
        self.x[5] = msg.twist.twist.linear.z
        euler_angles = np.array(euler_from_quaternion([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
            ]))
        # print "euler_angles", euler_angles
        self.x[6] = np.cos(euler_angles[0])
        self.x[7] = np.sin(euler_angles[0])
        self.x[8] = np.cos(euler_angles[1])
        self.x[9] = np.sin(euler_angles[1])
        self.x[10] = np.cos(euler_angles[2])
        self.x[11] = np.sin(euler_angles[2])
        
        self.x[12] = msg.twist.twist.angular.x
        self.x[13] = msg.twist.twist.angular.y
        self.x[14] = msg.twist.twist.angular.z

        # self.euler.data[0] = self.x[3]
        # self.euler.data[1] = self.x[4]

        # self.x_[0] = msg.pose.pose.position.x
        # self.x_[1] = msg.pose.pose.position.y
        # self.x_[2] = msg.pose.pose.position.z