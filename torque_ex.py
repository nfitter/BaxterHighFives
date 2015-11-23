#!/usr/bin/env python

# Hand-clapping code for the Rethink Robotics Baxter robot
# Based on Rethink Robotics's example, named below
"""
Baxter RSDK Joint Torque Example: joint springs
"""

import argparse
import rospy
from dynamic_reconfigure.server import (
    Server,
)
from std_msgs.msg import (
    Empty,
)
import baxter_interface
from baxter_interface import CHECK_VERSION
from sensor_msgs.msg import Imu
import time
import numpy as np
from scipy.signal import butter, lfilter

global start_time
flag = False
accel = []
vel_0 = {'right_s0': 0.0000000000000000, 'right_s1': 0.0000000000000000, 'right_w0': 0.0000000000000000, 'right_w1': 0.0000000000000000, 'right_w2': 0.0000000000000000, 'right_e0': 0.0000000000000000, 'right_e1': 0.0000000000000000}
count = 1
global pos0
global time0

def callback (msg):
  global flag
  global accel
  global start_time
  # Computer RMS acceleration from Baxter's wrist accelerometer
  rms_accel = (msg.linear_acceleration.x ** 2 + msg.linear_acceleration.y ** 2 + msg.linear_acceleration.z ** 2) ** (0.5)
  accel.append(rms_accel)
  # Experimentally, sampling rate of robot accelerometer is 200 Hz, Nyquist freqency 100 Hz
  accel_data = np.array(accel)
  b,a = butter(1,0.25,'highpass')
  y = lfilter(b,a,accel_data)

  if max(y)> 20:
    flag = True

class HighFiveArm(object):
    """
    Virtual Joint Springs class for torque example.
    @param limb: limb on which to run joint springs example
    @param reconfig_server: dynamic reconfigure server
    JointSprings class contains methods for the joint torque example allowing
    moving the limb to a neutral location, entering torque mode, and attaching
    virtual springs.
    """
    def __init__(self, limb):

        # control parameters
        self._rate = 1000.0  # Hz
        self._missed_cmds = 20.0  # Missed cycles before triggering timeout

        # create our limb instance
        self._limb = baxter_interface.Limb(limb)

        # initialize parameters
        self._Kp = dict()
        self._Kd = dict()
        self._Kf = dict()
        self._w = dict()

        # verify robot is enabled
        print("Getting robot state... ")
        self._rs = baxter_interface.RobotEnable(CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()
        print("Running. Ctrl-c to quit")

    def _update_parameters(self):
        '''for joint in self._limb.joint_names():
            self._Kp[joint] = 1
            self._Kd[joint] = 1'''
        # Proportional gains
        self._Kp['right_s0'] = 15
        self._Kp['right_s1'] = 15
        self._Kp['right_w0'] = 15
        self._Kp['right_w1'] = 25
        self._Kp['right_w2'] = 15
        self._Kp['right_e0'] = 15
        self._Kp['right_e1'] = 15
        # Derivative gains
        self._Kd['right_s0'] = 2
        self._Kd['right_s1'] = 2
        self._Kd['right_w0'] = 2
        self._Kd['right_w1'] = 2
        self._Kd['right_w2'] = 2
        self._Kd['right_e0'] = 2
        self._Kd['right_e1'] = 2
        # Feedforward "gains"
        self._Kf['right_s0'] = 0
        self._Kf['right_s1'] = 0
        self._Kf['right_w0'] = 0
        self._Kf['right_w1'] = 1
        self._Kf['right_w2'] = 0
        self._Kf['right_e0'] = 0
        self._Kf['right_e1'] = 0
        # Velocity filter weight
        self._w['right_s0'] = 0.1
        self._w['right_s1'] = 0.1
        self._w['right_w0'] = 0.1
        self._w['right_w1'] = 0.1
        self._w['right_w2'] = 0.1
        self._w['right_e0'] = 0.1
        self._w['right_e1'] = 0.1

    def _update_forces(self):
        global start_time
        global flag
        global count
        global pos0
        global time0

        """
        Calculates the current angular difference between the desired
        and the current joint positions applying a torque accordingly 
        using PD control.
        """
        # Define Baxter starting pose for hand-clapping interaction
        start_pose = {'right_s0': -1.1953545275939943, 'right_s1': -0.12693691005249025, 'right_w0': 1.34568464463501, 'right_w1': 1.8998352036254884, 'right_w2': 0.0, 'right_e0': 1.3265098848083496, 'right_e1': 2.0168012385681156}

        # get latest spring constants
        self._update_parameters()

        # create our command dict
        cmd = dict()

        # record current angles/velocities
        cur_pos = self._limb.joint_angles()
        pos1 = cur_pos
        #cur_vel = self._limb.joint_velocities()
        time1 = time.time()
        # angular velocity computed by change in angle over change in time
        comp_vel = dict()
        time_dict0 = dict()
        time_dict1 = dict()
        # make time into appropriate dictionary
        if count > 1:
            for joint in self._limb.joint_names():
                time_dict0[joint] = time0
                time_dict1[joint] = time1
            for joint in self._limb.joint_names():
                comp_vel[joint] = (pos1[joint] - pos0[joint]) / (time_dict1[joint] - time_dict0[joint])

        # identify amplitude and fequency of desired robot gripper movement
        amp = 3*0.175/2 #m
        freq = 1.00 #Hz

        # jump ahead in time if hand impact is felt
        #if flag:
        #    time_jump = (freq/2) - 2*(time.time() % freq)
        #    start_time = start_time + time_jump
        #    flag = False

        # find current time and use to calculate desired position, velocity
        time0 = time1
        pos0 = cur_pos
        elapsed_time = time.time() - start_time
        des_angular_displacement = -amp*np.sin(2*np.pi*freq*elapsed_time)
        des_angular_velocity = -amp*2*np.pi*freq*np.cos(2*np.pi*freq*elapsed_time)
        desired_pose = start_pose
        desired_pose['right_w1'] = 1.7341652787231447 + des_angular_displacement
        desired_velocity = {'right_s0': 0.0000000000000000, 'right_s1': 0.0000000000000000, 'right_w0': 0.0000000000000000, 'right_w1': 0.0000000000000000, 'right_w2': 0.0000000000000000, 'right_e0': 0.0000000000000000, 'right_e1': 0.0000000000000000}
        desired_velocity['right_w1'] = des_angular_velocity
        desired_feedforward = {'right_s0': 0.0000000000000000, 'right_s1': 0.0000000000000000, 'right_w0': 0.0000000000000000, 'right_w1': 0.0000000000000000, 'right_w2': 0.0000000000000000, 'right_e0': 0.0000000000000000, 'right_e1': 0.0000000000000000}
        desired_feedforward['right_w1'] = -des_angular_displacement

        # calculate current forces
        for joint in self._limb.joint_names():
            if count == 1:
                # For very start of robot motion, assume velocity 0, set smooth_vel to 0
                smooth_vel = {'right_s0': 0.0000000000000000, 'right_s1': 0.0000000000000000, 'right_w0': 0.0000000000000000, 'right_w1': 0.0000000000000000, 'right_w2': 0.0000000000000000, 'right_e0': 0.0000000000000000, 'right_e1': 0.0000000000000000}
                vel_0[joint] = smooth_vel[joint]
                # Torque to apply calculated with PD coltrol + feeforward term
                cmd[joint] = self._Kp[joint] * (desired_pose[joint] - cur_pos[joint]) + self._Kd[joint] * (desired_velocity[joint] - smooth_vel[joint]) + self._Kf[joint] * desired_feedforward[joint]
        smooth_vel = dict()
        for joint in self._limb.joint_names():
            if count > 1:
                # Compute smoothed version of current velocity
                smooth_vel[joint] = self._w[joint] * comp_vel[joint] + (1 - self._w[joint]) * vel_0[joint]
                vel_0[joint] = smooth_vel[joint]
                # Torque to apply calculated with PD coltrol + feeforward term
                cmd[joint] = self._Kp[joint] * (desired_pose[joint] - cur_pos[joint]) + self._Kd[joint] * (desired_velocity[joint] - smooth_vel[joint]) + self._Kf[joint] * desired_feedforward[joint]
        # command new joint torques
        count = count + 1
        self._limb.set_joint_torques(cmd)

    def move_to_start_position(self):
        # Define Baxter starting pose for hand-clapping interaction
        start_pose = {'right_s0': -1.2582477398254395, 'right_s1': 0.3033447004577637, 'right_w0': 1.290844831530762, 'right_w1': 1.7341652787231447, 'right_w2': -0.041033986029052734, 'right_e0': 1.5378157380981445, 'right_e1': 2.1682818411987306}

    def clean_shutdown(self):
        """
        Switches out of joint torque mode to exit cleanly
        """
        print("\nExiting example...")
        self._limb.exit_control_mode()
        if not self._init_state and self._rs.state().enabled:
            print("Disabling robot...")
            self._rs.disable()

def main():
    """RSDK Joint Torque Example: Joint Springs
    Moves the specified limb to a neutral location and enters
    torque control mode, attaching virtual springs (Hooke's Law)
    to each joint maintaining the start position.
    Run this example on the specified limb and interact by
    grabbing, pushing, and rotating each joint to feel the torques
    applied that represent the virtual springs attached.
    You can adjust the spring constant and damping coefficient
    for each joint using dynamic_reconfigure.
    """
    global start_time
    start_time = time.time()

    print("Initializing node... ")
    rospy.init_node("rsdk_joint_torque_springs_right")
    rospy.Subscriber ('/robot/accelerometer/right_accelerometer/state', 
    Imu, callback)

    # make high five arm object
    js = HighFiveArm('right')

    # set control rate
    #control_rate = rospy.Rate(js._rate)
    control_rate = rospy.Rate(1000)

    # for safety purposes, set the control rate command timeout.
    # if the specified number of command cycles are missed, the robot
    # will timeout and disable
    js._limb.set_command_timeout((1.0 / js._rate) * js._missed_cmds)

    # loop at specified rate commanding new joint torques
    while not rospy.is_shutdown():
        if not js._rs.state().enabled:
            rospy.logerr("Joint torque example failed to meet "
                         "specified control rate timeout.")
            break
        js._update_forces()
        control_rate.sleep()

    # register shutdown callback
    rospy.on_shutdown(js.clean_shutdown)
    js.move_to_neutral()

if __name__ == "__main__":
    main()