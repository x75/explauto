import numpy as np

import subprocess

from ..environment import Environment

from ...utils import bounds_min_max

import rospy
# from geometry_msgs.msg import Twist, Wrench, Point, Pose, Quaternion, Vector3
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class CopterMorseEnvironment(Environment):
    def __init__(self, m_ndims, s_ndims, m_mins, m_maxs, s_mins, s_maxs, sensor_transform):
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)
        rospy.init_node("coptermorseenvironment")
        # assert motor dim == control mode
    
        self.m_ndims = m_ndims
        self.s_ndims = s_ndims
        self.sensor_transform = sensor_transform
        
        self.x0 = np.zeros((self.s_ndims, ))
        self.x = self.x0.copy()
        self.current_context = np.dot(self.sensor_transform, self.x).flatten()

        # ros stuff
        self.r = rospy.Rate(10)
        self.pubs = {}
        self.subs = {}

        ctrl_mode = "attctrl"
        ctrl_type = Float32MultiArray
            
        self.attctrl = Float32MultiArray()
        self.attctrl.data = [0 for i in range(self.m_ndims)]
        
        self.pubs[ctrl_mode] = rospy.Publisher("/quad1/%s" % (ctrl_mode), ctrl_type, queue_size = 2)

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
        self.x[3] = np.cos(euler_angles[2])
        self.x[4] = np.sin(euler_angles[2])
    def cb_imu(self, msg):
        # print "imu", msg
        # self.x[]
        pass
                
    def compute_motor_command(self, ag_state):
        return bounds_min_max(ag_state, self.conf.m_mins, self.conf.m_maxs)

    def reset(self):
        self.x = self.x0.copy()
        
    def compute_sensori_effect(self, m):
        print("m", m)
        self.attctrl.data = m.tolist()
        self.pubs["attctrl"].publish(self.attctrl)
        self.r.sleep()
        print "self.x", self.x
        sensors = self.x.copy() # np.random.uniform(-1, 1, (self.s_ndims,))
        # print("%s, %s" % (self.__class__.__name__, sensors))
        self.current_context = sensors.copy()
        return sensors
