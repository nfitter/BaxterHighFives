#!/usr/bin/env python

# Hand-clapping code for the Rethink Robotics Baxter robot
# Based on Rethink Robotics's example, named below
"""
Baxter RSDK Joint Torque Example: joint springs
"""

import argparse
import os
import sys
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
from sensor_msgs.msg import (
    Imu, Image,
)
import time
import numpy as np
from scipy.signal import butter, lfilter
import cv2
import cv_bridge

global start_time
global rep_start
flag = 1
accelx = []
accely = []
accelz = []
vel_0 = {'right_s0': 0.0000000000000000, 'right_s1': 0.0000000000000000, 'right_w0': 0.0000000000000000, 'right_w1': 0.0000000000000000, 'right_w2': 0.0000000000000000, 'right_e0': 0.0000000000000000, 'right_e1': 0.0000000000000000}
count = 1
global pos0
global time0
global des_disp0
pause_start = time.time()
time_pause = 0

#####################################################
# Define experimental variables for trial

# identify whether the face will be animated (Boolean)
face_anim = True

# identify whether the robot will respond physically (Boolean)
phys_resp = True

# identify robot stiffness case (Boolean - True more stiff)
stiff_case = True

# identify amplitude and fequency of desired robot gripper movement (1.833, 2.667, 3.5)
freq = 1.833 #Hz
#####################################################

def callback (msg):
  global flag
  global accel
  global start_time
  global rep_start
  # Wait a reasonable length of hand-clapping cycle before listening
  if time.time() - rep_start < 0.1:
    flag = 1
  if time.time() - rep_start > 0.1:
    flag = 2
    '''# Computer RMS acceleration from Baxter's wrist accelerometer
    rms_accel = (msg.linear_acceleration.x ** 2 + msg.linear_acceleration.y ** 2 + msg.linear_acceleration.z ** 2) ** (0.5)
    accel.append(rms_accel)
    # Experimentally, sampling rate of robot accelerometer is 200 Hz, Nyquist freqency 100 Hz
    accel_data = np.array(accel)
    b,a = butter(1,[0.6, 0.7],'band')
    y = lfilter(b,a,accel_data)'''
    # Or actually, we should do the filtering on each channel separately
    # Store buffer of accelerometer data and convert it into a numpy data type
    accelx.append(msg.linear_acceleration.x)
    accely.append(msg.linear_acceleration.y)
    accelz.append(msg.linear_acceleration.z)
    accelx_data = np.array(accelx)
    accely_data = np.array(accely)
    accelz_data = np.array(accelz)
    # Fliter data with a highpass filter
    b,a = butter(1,0.25,'highpass')
    x_filt = lfilter(b,a,accelx_data)
    y_filt = lfilter(b,a,accely_data)
    z_filt = lfilter(b,a,accelz_data)
    # Compute square root of the sum of squares
    sqrsum_accel = (x_filt ** 2 + y_filt ** 2 + z_filt ** 2) ** (0.5)
    # See if result is high enough to indicate a hand impact
    if max(y)> 2.5:
      print(time.time() - rep_start)
      flag = 3
      accelx = []
      accely = []
      accelz = []
      print(True)
      rep_start = time.time()

def send_image(path):
    img = cv2.imread(path)
    msg = cv_bridge.CvBridge().cv2_to_imgmsg(img, encoding="bgr8")
    pub = rospy.Publisher('/robot/xdisplay', Image, latch=True)
    pub.publish(msg)

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

        # control parameters`
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
        if stiff_case:
	        # Proportional gains
	        self._Kp['right_s0'] = 20
	        self._Kp['right_s1'] = 20
	        self._Kp['right_w0'] = 20
	        self._Kp['right_w1'] = 30
	        self._Kp['right_w2'] = 20
	        self._Kp['right_e0'] = 20
	        self._Kp['right_e1'] = 20
	    else:
	    	# Proportional gains
	        self._Kp['right_s0'] = 25
	        self._Kp['right_s1'] = 25
	        self._Kp['right_w0'] = 25
	        self._Kp['right_w1'] = 35
	        self._Kp['right_w2'] = 25
	        self._Kp['right_e0'] = 25
	        self._Kp['right_e1'] = 25
        # Derivative gains
        self._Kd['right_s0'] = 3
        self._Kd['right_s1'] = 3
        self._Kd['right_w0'] = 3
        self._Kd['right_w1'] = 3
        self._Kd['right_w2'] = 3
        self._Kd['right_e0'] = 3
        self._Kd['right_e1'] = 3
        # Feedforward "gains"
        self._Kf['right_s0'] = 0
        self._Kf['right_s1'] = 0
        self._Kf['right_w0'] = 0
        self._Kf['right_w1'] = 1.5 # This was 2.0 before --> adjust to see if we can minimize overshoot
        self._Kf['right_w2'] = 0
        self._Kf['right_e0'] = 0
        self._Kf['right_e1'] = 0
        # Velocity filter weight
        self._w['right_s0'] = 0.075
        self._w['right_s1'] = 0.075
        self._w['right_w0'] = 0.075
        self._w['right_w1'] = 0.075
        self._w['right_w2'] = 0.075
        self._w['right_e0'] = 0.075
        self._w['right_e1'] = 0.075

    def _update_forces(self):
        global start_time
        global flag
        global count
        global pos0
        global time0
		global des_disp0
		global pause_start
		global time_pause
		global freq

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
        cur_vel = self._limb.joint_velocities()
        time1 = time.time() - start_time
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
                comp_vel[joint] = float(pos1[joint] - pos0[joint]) / float(time_dict1[joint] - time_dict0[joint])
            #print(pos1['right_w1'] - pos0['right_w1'])

        # hold still for some time if hand impact is felt and gripper is approaching human partner's hand
        if flag == 3 and vel_0['right_w1'] < 0:
        	T = 1.000/freq
        	# For robot stopping as response case
            time_pause = (T/2) - 2*((time.time() % freq)-(T/2))
            '''# For robot retreating slowly as response case
            time_pause = (T/2) - ((time.time() % freq) - (T/2))'''
            pause_start = time.time()

        # find current time and use to calculate desired position, velocity
        time0 = time1
        pos0 = cur_pos
        elapsed_time = time.time() - start_time
        if (time.time() - pause_start) < time_pause and phys_resp:
        	# For robot stopping as response case
        	des_angular_displacement = des_disp0
        	des_angular_velocity = 0
        	'''# For robot retreating slowly as result case
        	des_angular_velocity = (-des_disp0)/(time_pause)
        	des_angular_displacement = des_disp0 + (des_angular_velocity*(time.time() - pause_start))
			des_disp0 = des_angular_displacement'''        	
        else:
        	# Look up amplitude for commanded clapping frequency
        	# This could be changed to the best fit equation I found for continual clapping adjustment
        	# This, combined with the temporal difference learning from my quals, could let the robot adapt tempo in real time
        	amp_dict = {1.833:0.1725, 2.667:0.1125, 3.5:0.9}
        	amp = amp_dict[freq]
        	# Plug current time into characteristic trajectory equations, including phase shift so that robot starts out moving toward person
        	des_angular_displacement = -amp*np.sin(2*np.pi*freq*elapsed_time + (1/(4*freq)))
        	des_disp0 = des_angular_displacement
        	des_angular_velocity = -amp*2*np.pi*freq*np.cos(2*np.pi*freq*elapsed_time + (1/(4*freq)))
        desired_pose = start_pose
        desired_pose['right_w1'] = 1.7341652787231447 + (amp-0.1725)/2 + des_angular_displacement
        desired_velocity = {'right_s0': 0.0000000000000000, 'right_s1': 0.0000000000000000, 'right_w0': 0.0000000000000000, 'right_w1': 0.0000000000000000, 'right_w2': 0.0000000000000000, 'right_e0': 0.0000000000000000, 'right_e1': 0.0000000000000000}
        desired_velocity['right_w1'] = des_angular_velocity
        desired_feedforward = {'right_s0': 0.0000000000000000, 'right_s1': 0.0000000000000000, 'right_w0': 0.0000000000000000, 'right_w1': 0.0000000000000000, 'right_w2': 0.0000000000000000, 'right_e0': 0.0000000000000000, 'right_e1': 0.0000000000000000}
        desired_feedforward['right_w1'] = -des_angular_displacement

        # calculate current forces
        for joint in self._limb.joint_names():
            if count == 1:
                # For very start of robot motion, assume velocity 0, set smooth_vel to 0
                smooth_vel = cur_vel #{'right_s0': 0.0000000000000000, 'right_s1': 0.0000000000000000, 'right_w0': 0.0000000000000000, 'right_w1': 0.0000000000000000, 'right_w2': 0.0000000000000000, 'right_e0': 0.0000000000000000, 'right_e1': 0.0000000000000000}
                vel_0[joint] = smooth_vel[joint]
                # Torque to apply calculated with PD coltrol + feeforward term
                cmd[joint] = self._Kp[joint] * (desired_pose[joint] - cur_pos[joint]) + self._Kd[joint] * (desired_velocity[joint] - smooth_vel[joint]) + self._Kf[joint] * desired_feedforward[joint]
        smooth_vel = dict()
        for joint in self._limb.joint_names():
            if count > 1:
                # Compute smoothed version of current velocity
                smooth_vel[joint] = self._w[joint] * cur_vel[joint] + (1 - self._w[joint]) * vel_0[joint]
                vel_0[joint] = smooth_vel[joint]
                # Torque to apply calculated with PD coltrol + feeforward term
                cmd[joint] = self._Kp[joint] * (desired_pose[joint] - cur_pos[joint]) + self._Kd[joint] * (desired_velocity[joint] - smooth_vel[joint]) + self._Kf[joint] * desired_feedforward[joint]
        # record variables of interest to text doc
        f = open("torques.txt", "a")
        f.write(str(self._limb.joint_effort('right_s0')) + ',' + str(self._limb.joint_effort('right_s1')) + ',' + str(self._limb.joint_effort('right_w0')) + ',' + str(self._limb.joint_effort('right_w1')) + ',' + str(self._limb.joint_effort('right_w2')) + ',' + str(self._limb.joint_effort('right_e0')) + ',' + str(self._limb.joint_effort('right_e1')) + "\n")
        #f.write(str(elapsed_time) + ',' + str(cur_pos['right_w1']) + ',' + str(cur_vel['right_w1']) + ',' + str(vel_0['right_w1']) + ',' + str(desired_pose['right_w1']) + ',' + str(desired_velocity['right_w1']) + ',' + str(self._Kf['right_w1'] * desired_feedforward['right_w1']) + ','+ str(pos1['right_w1'] - pos0['right_w1']) + "\n")
        f.close()
        #print("%.6f" % time1)
        '''print(time1)
        print(cur_pos['right_w1'])
        print(cur_vel['right_w1'])
        print(vel_0['right_w1'])
        print(desired_pose['right_w1'])
        print(desired_velocity['right_w1'])'''
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
    global rep_start

    # Let user input experiment trial number
    parser = argparse.ArgumentParser(description = 'Input stimulus number.')
    parser.add_argument('integers', metavar = 'N', type = int, nargs = '+', help = 'an integer representing exp conds')
    trial_type = parser.parse_args()
    # Make lookup table of conditions based on this input, assign appropriate values to global variables controlling trial conditions

    # Start time
    start_time = time.time()
    rep_start = time.time()

    print("Initializing node... ")
    rospy.init_node("rsdk_joint_torque_springs_right")
    rospy.Subscriber ('/robot/accelerometer/right_accelerometer/state', 
    Imu, callback)

    # load face image on Baxter's screen
    # On Baxter base station
    send_image('/home/baxter/naomi_ws/src/baxter_examples/share/images/RestingFace.png')

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
                # animate face if impact is felt
        if face_anim:
	        if flag == 3:
	            send_image('/home/baxter/naomi_ws/src/baxter_examples/share/images/ReactingFace.png')
	        if flag == 2:
	            send_image('/home/baxter/naomi_ws/src/baxter_examples/share/images/RestingFace.png')
        js._update_forces()
        control_rate.sleep()

    # register shutdown callback
    rospy.on_shutdown(js.clean_shutdown)
    js.move_to_neutral()

if __name__ == "__main__":
    main()