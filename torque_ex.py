#!/usr/bin/env python

# Hand-clapping code for the Rethink Robotics Baxter robot
# Based on Rethink Robotics's example, 
# "Baxter RSDK Joint Torque Example: joint springs"

#####################################################
# Import necessary packages

import rospy
import argparse                             # package for allowing user input
import baxter_interface                     # package letting us access the Baxter limb API
from baxter_interface import CHECK_VERSION 
from sensor_msgs.msg import (
    Imu, Image,
)                                           # package letting us interface with Baxter's accelerometer and screen 
import time
import numpy as np
from scipy.signal import butter, lfilter    # package helping us filter accelerometer signal
import cv2                                  # package helping us with face animation
import cv_bridge                            # package helping us with face animation

#####################################################
# Define global variables to be accessible between functions

global start_time                           # overall start time of Baxter processes
global rep_start                            # timer starting after each hand impact
pause_start = time.time()                   # start time of pause behavior of Baxter's gripper after hand impact on advance
time_pause = 0                              # length of robot pause computed based on time of impact                       
vel_0 = {'right_s0': 0.00, 'right_s1': 0.00, 'right_w0': 0.00, 'right_w1': 0.00, 'right_w2': 0.00, 'right_e0': 0.00, 'right_e1': 0.00}
accelx = []                                 # x-wrist acceleration buffer
accely = []                                 # y-wrist acceleration buffer
accelz = []                                 # z-wrist acceleration buffer
loop_count = 1                              # variable allowing computation exception the first time through the code loop
state = 1                                   # variable controlling the state of the robot
                                            # state 1: not looking for hand impact, state 2: looking for hand impact, state 3: felt impact
                                            # the state variable is also used to control face rstoration to RestingFace

#####################################################
# Define global experimental variables for trial

# Identify whether the face will be animated (Boolean)
face_anim = True

# Identify whether the robot will respond physically (Boolean)
phys_resp = False

# Identify robot stiffness case (Boolean - True more stiff)
stiff_case = False

# Identify amplitude and fequency of desired robot gripper movement (1.833, 2.667, 3.5)
freq = 3.5 # Hz
global amp

#####################################################
# Callback function for processing data from robot's wrist accelerometer

def callback(msg):
    # Produces processed accelerometer reading for contact detection
    global start_time
    global rep_start
    global accelx
    global accely
    global accelz
    global state

    # Wait a reasonable length of hand-clapping cycle before listening
    if time.time()-rep_start < 0.1: # seconds
        # Allow for a brief pause in contact detection after hand contact to avoid button deboucing problem
        state = 1
    if time.time()-rep_start > 0.1: # seconds
        # Start looking for hand contact
        state = 2

        # Animate face back to RestingFace
        send_image('/home/baxter/naomi_ws/src/baxter_examples/share/images/RestingFace.png')

        # Do the filtering on each accelerometer axis separately
        # Store buffer of accelerometer data and convert it into a numpy data type
        accelx.append(msg.linear_acceleration.x)
        accely.append(msg.linear_acceleration.y)
        accelz.append(msg.linear_acceleration.z)
        accelx_data = np.array(accelx)
        accely_data = np.array(accely)
        accelz_data = np.array(accelz)

        # Fliter data with a bandpass filter
        b,a = butter(1,[0.5, 0.7],'band')
        x_filt = lfilter(b,a,accelx_data)
        y_filt = lfilter(b,a,accely_data)
        z_filt = lfilter(b,a,accelz_data)

        # Compute square root of the sum of squares
        sqrsum_accel = (x_filt**2 + y_filt**2 + z_filt**2) ** (0.5)

        # See if result is high enough to indicate a hand impact
        if max(sqrsum_accel) > 5.0:
            # Hand contact state
            state = 3

            # Animate face if contact felt
            send_image('/home/baxter/naomi_ws/src/baxter_examples/share/images/ReactingFace.png')
            
            # Empty buffers
            accelx = []
            accely = []
            accelz = []
            print(True)
            
            # Define new rep start time
            rep_start = time.time()

#####################################################
# Function for pulishing images to Baxter's screen

def send_image(path):
    # Load desired image on Baxter's screen
    img = cv2.imread(path)
    msg = cv_bridge.CvBridge().cv2_to_imgmsg(img, encoding="bgr8")
    pub = rospy.Publisher('/robot/xdisplay', Image, latch=True)
    pub.publish(msg)

#####################################################
# High Five Arm class for allowing Baxter to execute desired arm motion

class HighFiveArm(object):
    # Initializing high five arm class
    def __init__(self, limb):

        # Loop control parameters
        self._rate = 1000.0  # Hz
        self._missed_cmds = 20.0  # missed cycles before triggering timeout

        # Create our limb instance
        self._limb = baxter_interface.Limb(limb)

        # Initialize feedback control parameters
        self._Kp = dict()
        self._Kd = dict()
        self._Kf = dict()
        self._w = dict()

        # Verify robot is enabled
        print("Getting robot state... ")
        self._rs = baxter_interface.RobotEnable(CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()
        print("Running. Ctrl-c to quit")

    def _update_parameters(self):
        # Define feedback control parameters
        for joint in self._limb.joint_names():
            # Proportional gains
            if stiff_case:
                self._Kp[joint] = 20
            else:
                self._Kp[joint] = 30

            # Derivative gains
            self._Kd[joint] = 3

            # Feedforward "gains"
            self._Kf[joint] = 0

            # Vanishing memory filter weight (for finding smoothed velocity)
            self._w[joint] = 0.075

        # Overwrite the feedforward "gain" to nonzero for the one joint you actually want to move
        self._Kf['right_w1'] = 2.0 # <--play with this value to see if we can reduce overshoot

    def _update_forces(self):
        # Calculates the current angular difference between the desired
        # and the current joint positions applying a torque accordingly 
        # using PD control.
        global start_time
        global pause_start
        global time_pause
        global vel_0
        global loop_count
        global state
        global freq
        global amp
        
        # Define Baxter starting pose for hand-clapping interaction, obtained by posing robot and then querying joint positions
        start_pose = {'right_s0': -1.20, 'right_s1': -0.13, 'right_w0': 1.35, 'right_w1': 1.73, 'right_w2': 0.00, 'right_e0': 1.33, 'right_e1': 2.02}

        # Get latest feedback control constants
        self._update_parameters()

        # Initialized dictionary of torque values
        cmd = dict()

        # Record current angles/velocities
        cur_pos = self._limb.joint_angles()
        cur_vel = self._limb.joint_velocities()

        # Calculate amount of time to hold still if hand impact is felt and gripper is approaching human partner's hand
        if state == 3 and vel_0['right_w1'] <= 0:
            T = 1.000/freq

            # For robot stopping as response case
            time_pause = (T/2) - 2*((time.time() % freq)-(T/2))
            '''# For robot retreating slowly as response case
            time_pause = (T/2) - ((time.time() % freq) - (T/2))'''
            pause_start = time.time()

        # Find current time and use to calculate desired position, velocity
        elapsed_time = time.time() - start_time
        if (time.time() - pause_start) < time_pause and phys_resp:

            # For robot stopping as response case
            des_angular_displacement = des_disp0 # save desired displacement for next loop
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

            # Plug current time into characteristic sinusoidal trajectory equations, including phase shift so that robot starts out moving toward person
            des_angular_displacement = -amp*np.sin(2*np.pi*freq*elapsed_time + (1/(4*freq)))
            des_disp0 = des_angular_displacement # save desired displacement for next loop
            des_angular_velocity = -amp*2*np.pi*freq*np.cos(2*np.pi*freq*elapsed_time + (1/(4*freq)))
        
        # Define desired robot pose
        desired_pose = start_pose
        motion_center = start_pose['right_w1'] - amp + (amp-0.1725)/2 # far retreat (starting) postition minus ampltude to center plus factor to make hand-clap occur at same workspace position every time 
        desired_pose['right_w1'] = des_angular_displacement + motion_center
        
        # Define desired robot velocity
        desired_velocity = {'right_s0': 0.00, 'right_s1': 0.00, 'right_w0': 0.00, 'right_w1': 0.00, 'right_w2': 0.00, 'right_e0': 0.00, 'right_e1': 0.00}
        desired_velocity['right_w1'] = des_angular_velocity

        # Define something proportional to desired feedforward
        desired_feedforward = {'right_s0': 0.00, 'right_s1': 0.00, 'right_w0': 0.00, 'right_w1': 0.00, 'right_w2': 0.00, 'right_e0': 0.00, 'right_e1': 0.00}
        desired_feedforward['right_w1'] = -des_angular_displacement

        # Define average needed gravity compensation based on torque recording from stationary robot arm
        grav_comp = {'right_s0': -0.6020509915, 'right_s1': 7.4283286119, 'right_w0': 0.3976203966, 'right_w1': 1.7659150142, 'right_w2': -0.0344249292, 'right_e0': 19.748305949, 'right_e1': 1.1424249292}

        # Calculate torques to be applied this iteration
        smooth_vel = dict()
        for joint in self._limb.joint_names():
            if loop_count == 1:
                # For very start of robot motion, assume velocity 0, set smooth_vel to 0
                smooth_vel[joint] = cur_vel[joint]
                vel_0[joint] = smooth_vel[joint] # save smoothed velocity for next loop
                loop_count = loop_count + 1
            else:
                # Compute smoothed version of current velocity
                smooth_vel[joint] = self._w[joint] * cur_vel[joint] + (1 - self._w[joint]) * vel_0[joint]
            # Save current smoothed velocity for use in the next iteration
            vel_0[joint] = smooth_vel[joint]
            # Torque to apply calculated with PD coltrol + feeforward term
            cmd[joint] = self._Kp[joint]*(desired_pose[joint]-cur_pos[joint]) + self._Kd[joint]*(desired_velocity[joint]-smooth_vel[joint]) + self._Kf[joint]*desired_feedforward[joint] + 0.5*grav_comp[joint]

        # Record variables of interest to text doc
        #f = open("torques.txt", "a")
        #f.write(str(elapsed_time) + ',' + str(cur_pos['right_w1']) + ',' + str(cur_vel['right_w1']) + ',' + str(vel_0['right_w1']) + ',' + str(desired_pose['right_w1']) + ',' + str(desired_velocity['right_w1']) + ',' + str(self._Kf['right_w1'] * desired_feedforward['right_w1']) + "\n")
        #f.close()

        # Command new joint torques
        self._limb.set_joint_torques(cmd)

    def clean_shutdown(self):
        #Switches out of joint torque mode to exit cleanly
        print("\nExiting example...")
        self._limb.exit_control_mode()
        if not self._init_state and self._rs.state().enabled:
            print("Disabling robot...")
            self._rs.disable()

#####################################################
# Main loop of code

def main():
    # Moves Baxter's arm in a hand-clapping trajectory with varying 
    # reactivity, tempo, and stiffness based on user input
    global start_time
    global rep_start

    # Let user input experiment trial number
    #parser = argparse.ArgumentParser(description = 'Input stimulus number.')
    #parser.add_argument('integers', metavar = 'N', type = int, nargs = '+', help = 'an integer representing exp conds')
    #trial_type = parser.parse_args()
    # Make lookup table of conditions based on this input, assign appropriate values to global variables controlling trial conditions

    # Start time
    start_time = time.time()
    rep_start = time.time()

    print("Initializing node... ")
    rospy.init_node("rsdk_joint_torque_springs_right")
    rospy.Subscriber ('/robot/accelerometer/right_accelerometer/state', 
    Imu, callback)

    # Load face image on Baxter's screen
    send_image('/home/baxter/naomi_ws/src/baxter_examples/share/images/RestingFace.png')

    # Make high five arm object
    js = HighFiveArm('right')

    # Set control rate
    control_rate = rospy.Rate(1000)

    # For safety purposes, set the control rate command timeout.
    # If the specified number of command cycles are missed, the robot
    # will timeout and disable.
    js._limb.set_command_timeout((1.0 / js._rate) * js._missed_cmds)

    # Loop at specified rate commanding new joint torques
    while not rospy.is_shutdown():
        if not js._rs.state().enabled:
            rospy.logerr("Joint torque example failed to meet "
                         "specified control rate timeout.")
            break
        js._update_forces()
        control_rate.sleep()

if __name__ == "__main__":
    main()