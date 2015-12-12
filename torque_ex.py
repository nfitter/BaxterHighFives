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
del_t = [100]                               # array for storing time between felt hand impacts           
vel_0 = {'right_s0': 0.00, 'right_s1': 0.00, 'right_w0': 0.00, 'right_w1': 0.00, 'right_w2': 0.00, 'right_e0': 0.00, 'right_e1': 0.00}
des_disp0 = 0                               # providing initial value for desired displacement saved across loops
accelx = []                                 # x-wrist acceleration buffer
accely = []                                 # y-wrist acceleration buffer
accelz = []                                 # z-wrist acceleration buffer
loop_count = 1                              # variable allowing computation exception the first time through the code loop
state = 1                                   # variable controlling the state of the robot
                                            # state 1: not looking for hand impact, state 2: looking for hand impact, state 3: felt impact
                                            # the state variable is also used to control face rstoration to RestingFace

#####################################################
# Initialize global experimental variables for trial

global trial_cond                           # identifies number of the relevant set of trial conditions
global face_anim                            # identifies whether the face will be animated (Boolean)
global phys_resp                            # identifies whether the robot will respond physically (Boolean)
global stiff_case                           # identifies robot stiffness case (Boolean - True more stiff)
global freq                                 # identifies amplitude and fequency of desired robot gripper movement (between 1 Hz and 4.33 Hz)

#####################################################
# Callback function for processing data from robot's wrist accelerometer

def callback(msg):
    # Produces processed accelerometer reading for contact detection
    global start_time
    global rep_start
    global del_t
    global accelx
    global accely
    global accelz
    global state
    # global freq <-- uncomment if we want to try active tempo adjustment

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

        # Fliter data with a highpass filter
        b,a = butter(1,[0.6],'highpass')
        x_filt = lfilter(b,a,accelx_data)
        y_filt = lfilter(b,a,accely_data)
        z_filt = lfilter(b,a,accelz_data)

        # Compute square root of the sum of squares
        sqrsum_accel = (x_filt**2 + y_filt**2 + z_filt**2) ** (0.5)

        elapsed_time = time.time() - start_time
        # Temporarily print out result to examine it
        f = open("test_accel", "a")
        f.write(str(elapsed_time) + ',' + str(sqrsum_accel[-1]) + str("\n"))
        f.close()

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

            '''# We could also use the equatior proposed in my quals to actively update clapping frequency
            del_t.append(time.time()-rep_start)
            period_pred = 2*del_t[-1] - del_t[-2]
            freq = 1.000/period_pred'''
            
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
                self._Kp[joint] = 30
            else:
                self._Kp[joint] = 15

            # Derivative gains
            self._Kd[joint] = 3

            # Feedforward "gains"
            self._Kf[joint] = 0

            # Vanishing memory filter weight (for finding smoothed velocity)
            self._w[joint] = 0.075

        # Overwrite the feedforward "gain" to nonzero for the one joint you actually want to move
        self._Kf['right_w1'] = 1.0 # <--play with this value to see if we can reduce overshoot

        # Overwrite w1 proportional gain to make it a little stiffer
        self._Kp['right_w1'] = 35

    def _update_joint_angles(self):
        # Use this to move arm gently to starting position for hand-clapping
        global freq

        # We can also find these amplitude of hand movement form the best fit equation I found in my old freq vs amplitude plot
        char_amp = 0.2575*np.exp(-0.4286*freq)
        amp = (3*char_amp)/2 # the above gives peak-to-peak amp, so divide by 2 and multiply by 3 to convert to correct distance in cartesian space
        
        # Define starting pose and make minor adjustment based on frequency of this rep
        start_pose = {'right_s0': -1.20, 'right_s1': 0.00, 'right_w0': 1.35, 'right_w1': 1.85, 'right_w2': 0.00, 'right_e0': 1.33, 'right_e1': 2.02}
        amplitude_compensation = (amp-0.2625) # take difference between current ampliture and max possible amplitude to make hand-clap occur at same joint angle every time 
        start_pose['right_w1'] = start_pose['right_w1'] + amplitude_compensation

        # Command move to initial pose
        self._limb.move_to_joint_positions(start_pose,timeout=5)
        end_positions = self._limb.joint_angles()
        thresh = 0.008726646

        # Iterate until arm close enough
        for key in start_pose.keys(): 
            diff = abs(start_pose[key] - end_positions[key])
            if diff > thresh: 
                print key, diff
                return False
            return True

    def _update_forces(self):
        # Calculates the current angular difference between the desired
        # and the current joint positions applying a torque accordingly 
        # using PD control.
        global start_time
        global pause_start
        global time_pause
        global vel_0
        global des_disp0
        global loop_count
        global state
        global trial_cond
        global freq
        
        # Define Baxter starting pose for hand-clapping interaction, obtained by posing robot and then querying joint positions
        start_pose = {'right_s0': -1.20, 'right_s1': -0.10, 'right_w0': 1.35, 'right_w1': 1.85, 'right_w2': 0.00, 'right_e0': 1.03, 'right_e1': 2.02}

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

        '''# Look up amplitude for commanded clapping frequency
        amp_dict = {1.000:0.2516, 1.833:0.1760, 2.667:0.1232, 3.500:0.0862, 4.333:0.0603}
        amp = amp_dict[freq]'''
        # We can also find these amplitudes form the best fit equation I found in my old freq vs amplitude plot
        char_amp = 0.2575*np.exp(-0.4286*freq)
        amp = (3*char_amp)/2 # the above gives peak-to-peak amp, so divide by 2 and multiply by 3 to convert to correct distance in cartesian space

        if (time.time() - pause_start) < time_pause and phys_resp:

            # For robot stopping as response case
            des_angular_displacement = des_disp0 # save desired displacement for next loop
            des_angular_velocity = 0
            '''# For robot retreating slowly as result case
            des_angular_velocity = (-des_disp0)/(time_pause)
            des_angular_displacement = des_disp0 + (des_angular_velocity*(time.time() - pause_start))
            des_disp0 = des_angular_displacement'''         
        else:

            # Plug current time into characteristic sinusoidal trajectory equations, including phase shift so that robot starts out moving toward person
            des_angular_displacement = amp*np.sin(2*np.pi*freq*elapsed_time + 2*np.pi*freq*(1/(4*freq))) - amp
            des_disp0 = des_angular_displacement # save desired displacement for next loop
            des_angular_velocity = amp*2*np.pi*freq*np.cos(2*np.pi*freq*elapsed_time + 2*np.pi*freq*(1/(4*freq)))
        
        # Define desired robot pose
        desired_pose = start_pose
        amplitude_compensation = (amp-0.2625) # take difference between current ampliture and max possible amplitude to make hand-clap occur at same joint angle every time 
        desired_pose['right_w1'] = desired_pose['right_w1'] + des_angular_displacement + amplitude_compensation
        
        # Define desired robot velocity
        desired_velocity = {'right_s0': 0.00, 'right_s1': 0.00, 'right_w0': 0.00, 'right_w1': 0.00, 'right_w2': 0.00, 'right_e0': 0.00, 'right_e1': 0.00}
        desired_velocity['right_w1'] = des_angular_velocity

        # Define something proportional to desired feedforward
        desired_feedforward = {'right_s0': 0.00, 'right_s1': 0.00, 'right_w0': 0.00, 'right_w1': 0.00, 'right_w2': 0.00, 'right_e0': 0.00, 'right_e1': 0.00}
        desired_feedforward['right_w1'] = -des_angular_displacement

        # Define average needed gravity compensation based on torque recording from stationary robot arm
        grav_comp = {'right_s0': -0.6020509915, 'right_s1': 7.4283286119, 'right_w0': 0.3976203966, 'right_w1': 0.0000000000, 'right_w2': -0.0344249292, 'right_e0': 19.748305949, 'right_e1': 1.1424249292}

        # Calculate torques to be applied this iteration
        smooth_vel = dict()
        for joint in self._limb.joint_names():
            if loop_count == 1:
                # For very start of robot motion, assume current velocity from Rethink's SDK is a good estimate
                smooth_vel[joint] = cur_vel[joint]
                vel_0[joint] = smooth_vel[joint] # save smoothed velocity for next loop
                loop_count = loop_count + 1
            else:
                # Compute smoothed version of current velocity offered by Rethink's SDK
                smooth_vel[joint] = self._w[joint] * cur_vel[joint] + (1 - self._w[joint]) * vel_0[joint]

            # Save current smoothed velocity for use in the next iteration
            vel_0[joint] = smooth_vel[joint]

            # Torque to apply calculated with PD coltrol + feeforward term
            cmd[joint] = self._Kp[joint]*(desired_pose[joint]-cur_pos[joint]) + self._Kd[joint]*(desired_velocity[joint]-smooth_vel[joint]) + self._Kf[joint]*desired_feedforward[joint] + 0.5*grav_comp[joint]

        # Record variables of interest to text doc
        #file_name = "output" + str(trial_cond)
        #f = open(file_name, "a")
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
    global face_anim
    global phys_resp
    global stiff_case
    global freq
    global trial_cond

    # Start time
    start_time = time.time()
    rep_start = time.time()

    # Let user input experiment trial number
    parser = argparse.ArgumentParser(description = 'Input stimulus number.')
    parser.add_argument('integers', metavar = 'N', type = int, nargs = '+', help = 'an integer representing exp conds')
    foo = parser.parse_args()
    trial_cond = (foo.integers[0])
    # Access lookup table of conditions based on this input, assign appropriate values to global variables controlling trial conditions
    trial_stimuli = {1:[False,False,False,1.833], 2:[False,False,False,2.667], 3:[False,False,False,3.500],
                    4:[True,False,False,1.833], 5:[True,False,False,2.667], 6:[True,False,False,3.500],
                    7:[False,True,False,1.833], 8:[False,True,False,2.667], 9:[False,True,False,3.500],
                    10:[False,False,True,1.833], 11:[False,False,True,2.667], 12:[False,False,True,3.500],
                    13:[True,True,False,1.833], 14:[True,True,False,2.667], 15:[True,True,False,3.500],
                    16:[False,True,True,1.833], 17:[False,True,True,2.667], 18:[False,True,True,3.500],
                    19:[True,False,True,1.833], 20:[True,False,True,2.667], 21:[True,False,True,3.500],
                    22:[True,True,True,1.833], 23:[True,True,True,2.667], 24:[True,True,True,3.500]}
    face_anim = trial_stimuli[trial_cond][0]
    phys_resp = trial_stimuli[trial_cond][1]
    stiff_case = trial_stimuli[trial_cond][2]
    freq = trial_stimuli[trial_cond][3] # Hz

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

    # Move robot to starting pose
    js._update_joint_angles()

    # For safety purposes, set the control rate command timeout.
    # If the specified number of command cycles are missed, the robot
    # will timeout and disable.
    js._limb.set_command_timeout((1.0 / js._rate) * js._missed_cmds)

    # Reinitialize start time to match desired trajectory
    start_time = time.time()
    rep_start = time.time()

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