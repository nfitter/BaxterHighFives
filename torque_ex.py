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
global time_delay                           # difference in start time of code vs motion object
global rep_start                            # timer starting after each hand impact
pause_start = time.time()                   # start time of pause behavior of Baxter's gripper after hand impact on advance
time_pause = 0                              # length of robot pause computed based on time of impact      
# del_t = [100]                             # array for storing time between felt hand impacts           <-- uncomment if we want to try active tempo adjustment
vel_0 = {'right_s0': 0.00, 'right_s1': 0.00, 'right_w0': 0.00, 'right_w1': 0.00, 'right_w2': 0.00, 'right_e0': 0.00, 'right_e1': 0.00}
des_disp0 = 0                               # providing initial value for desired displacement saved across loops
global coeffs                               # keeps track of coeffs of cubic for robot retreat behavior
accelx = []                                 # x-wrist acceleration buffer
loop_count = 1                              # variable allowing computation exception the first time through the code loop
cback_loop_count = 0                        # variable for keeping track of how many hand contacts have happened
pause_hand = False                          # variable for responsively changing hand trajectory
update_face = False                         # toggle for only updating the face when necessary
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
    # Callback loop produces processed accelerometer reading for contact detection
    global start_time
    global rep_start
    # global del_t <-- uncomment if we want to try active tempo adjustment
    global accelx
    global cback_loop_count
    global pause_hand
    global update_face
    global state
    global trial_cond
    global freq

    # Wait half of hand-clapping period (T) plus a bit before listening for new hand impact, to avoid button deboucing problem
    T = 1.000/freq

    if time.time()-rep_start < (T/2) + .05: # seconds
        # Allow for a brief pause in contact detection by staying in state 1 (not looking for hand impact)
        state = 1
        
    else:
        # Now start looking for hand contact in state 2 (looking for hand impact)
        state = 2

        # After doing the filtering on each accelerometer axis separately, we found that 1st order highpass Butterworth works well
        # There weren't meaningful peaks in y or z accelerometer axis recordings, so we just save buffer of data for x-axis below
        accelx.append(msg.linear_acceleration.x)

        # Fliter data with a highpass filter
        b,a = butter(1,0.5,'highpass')
        x_filt = lfilter(b,a,accelx)

        # Record the accelerations reported by robot
        elapsed_time = time.time() - start_time
        file_name = 'accel_recording' + str(trial_cond) + '.csv' 
        f = open(file_name, "a")
        f.write(str(elapsed_time) + ',' + str(msg.linear_acceleration.x) + ',' + str(msg.linear_acceleration.y) + ',' + str(msg.linear_acceleration.z) + str("\n"))
        f.close()

        # Look up threshold for commanded clapping frequency
        thresh_dict = {1.000:4.0, 1.833:5.0, 2.667:5.5, 3.500:7.0, 4.333:7.5}
        thresh = thresh_dict[freq]
        # ---> After this experiment, I should try to fit a model for this to make clapping tempo change possible, along with other things in lookup tables <---

        # See if result is high enough to indicate a hand impact
        if max(x_filt) > thresh:
            # State change to hand contact state
            state = 3
            
            # Empty buffers
            accelx = []

            # Set Boolean variables to allow other loops to switch expressions and physically react
            pause_hand = True
            update_face = True
            cback_loop_count = cback_loop_count + 1

            # Print nice output to give me feedback in real time (evidence of contact and timing of contact)
            print('________________ Contact felt! ________________')
            print('time between this and last impact:' + str(time.time()-rep_start))
            print('hand contact number:' + str(cback_loop_count))

            '''# We could also use the equatior proposed in my quals to actively update clapping frequency
            del_t.append(time.time()-rep_start)
            period_pred = 2*del_t[-1] - del_t[-2]
            freq = 1.000/period_pred'''
            
            # Reset repetition start time to current time
            rep_start = time.time()

#####################################################
# Function for pulishing images to Baxter's screen

def send_image(path):
    # Load desired image on Baxter's screen, as in Rethink's example code
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
        global stiff_case

        # Define feedback control parameters
        for joint in self._limb.joint_names():
            # Proportional gains
            if stiff_case:
                self._Kp[joint] = 60
            else:
                self._Kp[joint] = 15

            # Derivative gains
            if stiff_case:
                self._Kd[joint] = 3
            else:
                self._Kd[joint] = 3

            # Feedforward "gains"
            self._Kf[joint] = 0

            # Vanishing memory filter weight (for finding smoothed velocity)
            self._w[joint] = 0.075

        # In all cases, overwrite the w1 proportional gain to a bit higher to track trajectory well
        self._Kp['right_w1'] = 30

        # Overwrite the feedforward "gain" to nonzero for the one joint you actually want to move
        self._Kf['right_w1'] = 0.5

    def _update_joint_angles(self):
        # Use this to move arm gently to starting position for hand-clapping
        global freq

        # Look up amplitude for commanded clapping frequency
        #amp_dict = {1.000:0.2492, 1.833:0.1760, 2.667:0.1210, 3.500:0.0862, 4.333:0.0603}
        # Hand-dialed amplitudes below to make motion a little smaller for the experiment
        amp_dict = {1.000:0.213, 1.833:0.130, 2.667:0.070}
        amp = amp_dict[freq]
        # ---> if I update amplitudes here, need to update another place, too, and also in one if statement for biggest amplitude
        
        # Define starting pose and make minor adjustment based on frequency of this rep
        start_pose = {'right_s0': -0.8620972018066407, 'right_s1': 0.35665053277587894, 'right_w0': 1.1696603494262696, 'right_w1': 1.6193157223693849, 'right_w2': -0.02070874061279297, 'right_e0': 1.5132720455200197, 'right_e1': 1.9381847232788088}
        # ---> if I update start_pose here, need to update another place, too, and also update the grav comp feedforward torques <---
        amplitude_compensation = 0.9*2*(amp-0.213) # take difference between current ampliture and max possible amplitude to make hand-clap occur at same joint angle every time 
         # (leading multiplier is hand-tuned to fit our motion expectations)
        start_pose['right_w1'] = start_pose['right_w1'] + amplitude_compensation

        # Define tomeout for motion and grab current arm joint angles
        self._limb.move_to_joint_positions(start_pose,timeout=100) # we want to give it time to get all the way there
        end_positions = self._limb.joint_angles()
        thresh = 0.008726646

        # Iterate until arm is close enough to desired position
        for key in start_pose.keys(): 
            diff = abs(start_pose[key] - end_positions[key])
            if diff > thresh: 
                print key, diff
                return False
            return True

    def _update_end_pos(self):
        # Use this to move arm gently to far retreat position at end of trial
        
        # Define starting pose (same as ending pose)
        start_pose = {'right_s0': -0.8620972018066407, 'right_s1': 0.35665053277587894, 'right_w0': 1.1696603494262696, 'right_w1': 1.6193157223693849, 'right_w2': -0.02070874061279297, 'right_e0': 1.5132720455200197, 'right_e1': 1.9381847232788088}
        # ---> if I update start_pose here, need to update another place, too, and also update the grav comp feedforward torques <---

        # Define tomeout for motion and grab current arm joint angles
        self._limb.move_to_joint_positions(start_pose,timeout=100) # we want to give it time to get all the way there
        end_positions = self._limb.joint_angles()
        thresh = 0.008726646

        # Iterate until arm is close enough to desired position
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
        global time_delay
        global pause_start
        global time_pause
        global vel_0
        global des_disp0
        global coeffs
        global loop_count
        global pause_hand
        global state
        global trial_cond
        global phys_resp
        global freq
        
        # Define Baxter starting pose for hand-clapping interaction, obtained by posing robot and then querying joint positions
        start_pose = {'right_s0': -0.8620972018066407, 'right_s1': 0.35665053277587894, 'right_w0': 1.1696603494262696, 'right_w1': 1.6193157223693849, 'right_w2': -0.02070874061279297, 'right_e0': 1.5132720455200197, 'right_e1': 1.9381847232788088}
        # ---> if I update start_pose here, need to update another place, too, and also update the grav comp feedforward torques <---

        # Get latest feedback control constants
        self._update_parameters()

        # Initialized dictionary of torque values
        cmd = dict()

        # Record current joint angles/velocities
        cur_pos = self._limb.joint_angles()
        cur_vel = self._limb.joint_velocities()

        # Find current time and use to calculate desired position, velocity
        published_time = time.time() - start_time
        elapsed_time = time.time() - start_time - time_delay

        # Look up amplitude for commanded clapping frequency
        #amp_dict = {1.000:0.2492, 1.833:0.1760, 2.667:0.1210, 3.500:0.0862, 4.333:0.0603}
        # Hand-dialed amplitudes below to make motion a little smaller for the experiment
        amp_dict = {1.000:0.213, 1.833:0.130, 2.667:0.070}
        amp = amp_dict[freq]
        # ---> if I update amplitudes here, need to update another place, too, and also in one if statement for biggest amplitude
        '''# We can also find these amplitudes form the best fit equation I found in my old freq vs amplitude plot
        char_amp = 0.2575*np.exp(-0.4286*freq)
        # From the above equation, we know the desired chord length. Then, we can use 2*invsin((1/(2*radius)*chord length) to get total joint angle peak-to-peak for motion. 1/2 that is the amp we need for the motion eqautions below.
        amp = (2*np.arcsin((1/(2*0.34)*char_amp))/2
        # This eqn may need to be adjusted based on overshoot that Baxter exhibits at high tempos. ---> Fit new curve accordingly after experiment <---'''

        # Since I still haven't decided whether robot should stay still or retreat slowly as physical response, set Boolean to switch this
        stop_hand = False # <--- I can change this and show to Alex (True makes hand stop, False makes hand slow retreat)

        # Loop calculates amount of time to hold still/retreat with const veloctiy if hand impact is felt and gripper is approaching human partner's hand
        if pause_hand and vel_0['right_w1'] <= 0:
            T = 1.000/freq

            if cur_pos['right_w1'] < start_pose['right_w1'] + 0.9*2*(amp-0.213):
                if stop_hand:
                    # Compute length of motion stop for robot stopping case, impact felt closer to person
                    time_pause = (T/2) - 2*((elapsed_time % T)-(T/4))
                else:
                    # Compute length of slow retreat for robot retreating slowly case, impact felt closer to person
                    time_pause = T - (elapsed_time % T)
            else:
                if stop_hand:
                    # Compute length of motion stop for robot stopping case, impact felt closer to robot
                    time_pause = (T/2) - 2*(elapsed_time % T)
                else:
                    # Compute length of slow retreat for robot retreating slowly case, impact felt closer to robot
                    time_pause = T - (elapsed_time % T)

            # Determine coefficients for later cubic spline
            coeffs = [(2*des_disp0)/(time_pause ** 3), (3*-des_disp0)/(time_pause ** 2)]

            # Start timer for physical robot response, set Boolean so code knows physical response behavior parameters have already been generated
            pause_start = time.time()
            pause_hand = False

        # Loop updates the desired robot joint displacement from start position and joint velocity
        if (time.time() - pause_start) < time_pause and phys_resp:
            if stop_hand:
                # Update values for robot stopping case if still within calculated reaction time
                des_angular_displacement = des_disp0 # save desired displacement for next loop
                des_angular_velocity = 0
            else:
                # Update values for robot retreating slowly case if still within calculated reaction time
                # We can use a cubic spline for connecting data points, here assuming velocity is ~0 at start and end of this period
                des_angular_displacement = coeffs[0]*((time.time() - pause_start) ** 3) + coeffs[1]*((time.time() - pause_start) ** 2) + des_disp0
                des_angular_velocity = coeffs[0]*3*((time.time() - pause_start) ** 2) + coeffs[1]*2*(time.time() - pause_start)
        else:
            # If robot reaction time is over and/or physical response disabled...
            # Plug current time into characteristic sinusoidal trajectory equations, including phase shift so that robot starts out moving toward person
            des_angular_displacement = amp*np.sin(2*np.pi*freq*elapsed_time + 2*np.pi*freq*(1/(4*freq))) - amp
            des_disp0 = des_angular_displacement # save desired displacement for next loop
            des_angular_velocity = amp*2*np.pi*freq*np.cos(2*np.pi*freq*elapsed_time + 2*np.pi*freq*(1/(4*freq)))
        
        # Define desired robot joint angles (adding calculated displacement to centered start pose of robot)
        desired_pose = start_pose
        amplitude_compensation = 0.9*2*(amp-0.213) # take difference between current ampliture and max possible amplitude to make hand-clap occur at same joint angle every time 
        # (leading multiplier is hand-tuned to fit our motion expectations)
        desired_pose['right_w1'] = desired_pose['right_w1'] + des_angular_displacement + amplitude_compensation
        
        # Define desired robot velocity
        desired_velocity = {'right_s0': 0.00, 'right_s1': 0.00, 'right_w0': 0.00, 'right_w1': 0.00, 'right_w2': 0.00, 'right_e0': 0.00, 'right_e1': 0.00}
        desired_velocity['right_w1'] = des_angular_velocity

        # Define something proportional to desired feedforward (we know it's proportional to the negative position)
        desired_feedforward = {'right_s0': 0.00, 'right_s1': 0.00, 'right_w0': 0.00, 'right_w1': 0.00, 'right_w2': 0.00, 'right_e0': 0.00, 'right_e1': 0.00}
        desired_feedforward['right_w1'] = -des_angular_displacement

        # Define average needed gravity compensation based on torque recording from stationary robot arm
        grav_comp = {'right_s0': 0.0000000000, 'right_s1': -11.5113408, 'right_w0': 0.0000000000, 'right_w1': -2.21*np.cos(cur_pos['right_w1']) , 'right_w2': 0.0000000000, 'right_e0': 20.9088208, 'right_e1': -11.5113408}
        # Note: In limb object output, sign is ***wrong*** for right_s1, right_e1!?!? Maybe others, too...
        
        # Calculate torques to be applied this iteration
        smooth_vel = dict()
        for joint in self._limb.joint_names():
            if loop_count == 1:
                # For very start of robot motion, assume current velocity from Rethink's SDK is a good estimate
                smooth_vel[joint] = cur_vel[joint]
                loop_count = loop_count + 1
            else:
                # Compute smoothed version of current velocity offered by Rethink's SDK (limited because of coarse encoder spacing, but at least doesn't make motion jerky after smoothed)
                smooth_vel[joint] = self._w[joint] * cur_vel[joint] + (1 - self._w[joint]) * vel_0[joint]

            # Save current smoothed velocity for use in the next iteration
            vel_0[joint] = smooth_vel[joint]

            # Compute torque to apply calculated with PD coltrol + feedforward terms (gravity comp and ideal torques)
            if (time.time() - pause_start) < time_pause:
                # Disable ideal feedforward torque if some physical response is going on (different motion model)
                cmd[joint] = self._Kp[joint]*(desired_pose[joint]-cur_pos[joint]) + self._Kd[joint]*(desired_velocity[joint]-smooth_vel[joint]) + 0.14*grav_comp[joint]
            else:
                # Or don't any other time
                cmd[joint] = self._Kp[joint]*(desired_pose[joint]-cur_pos[joint]) + self._Kd[joint]*(desired_velocity[joint]-smooth_vel[joint]) + self._Kf[joint]*desired_feedforward[joint] + 0.14*grav_comp[joint]

        # Record variables of interest to text doc
        file_name = "output" + str(trial_cond) + '.csv'
        cur_torques = self._limb.joint_efforts()
        f = open(file_name, "a")
        # Uncomment next line to record just torques being exerted by Baxter
        #f.write(str(published_time) + ',' + str(cur_torques['right_s0']) + ',' + str(cur_torques['right_s1']) + ',' + str(cur_torques['right_w0']) + ',' + str(cur_torques['right_w1']) + ',' + str(cur_torques['right_w2']) + ',' + str(cur_torques['right_e0']) + ',' + str(cur_torques['right_s1']) + "\n")
        # Uncomment next line to record only actual and desired positions
        #f.write(str(published_time) + ',' + str(cur_pos['right_w1']) + ',' + str(desired_pose['right_w1']) + "\n")
        # ---> Uncomment next line to record everything for actual trials <---
        f.write(str(published_time) + ',' + str(time.time()) + ',' + str(face_anim) + ',' + str(phys_resp) + ',' + str(stiff_case) + ',' + str(freq) + ',' + 
                                            str(cur_pos['right_s0']) + ',' + str(cur_pos['right_s1']) + ',' + str(cur_pos['right_w0']) + ',' + str(cur_pos['right_w1']) + ',' + str(cur_pos['right_w2']) + ',' + str(cur_pos['right_e0']) + ',' + str(cur_pos['right_e1']) + ',' + 
                                            str(vel_0['right_s0']) + ',' + str(vel_0['right_s1']) + ',' + str(vel_0['right_w0']) + ',' + str(vel_0['right_w1']) + ',' + str(vel_0['right_w2']) + ',' + str(vel_0['right_e0']) + ',' + str(vel_0['right_e1']) + ',' + 
                                            str(desired_pose['right_s0']) + ',' + str(desired_pose['right_s1']) + ',' + str(desired_pose['right_w0']) + ',' + str(desired_pose['right_w1']) + ',' + str(desired_pose['right_w2']) + ',' + str(desired_pose['right_e0']) + ',' + str(desired_pose['right_e1']) + ',' + 
                                            str(desired_velocity['right_s0']) + ',' + str(desired_velocity['right_s1']) + ',' + str(desired_velocity['right_w0']) + ',' + str(desired_velocity['right_w1']) + ',' + str(desired_velocity['right_w2']) + ',' + str(desired_velocity['right_e0']) + ',' + str(desired_velocity['right_e1']) + ',' + 
                                            str(self._Kf['right_s0'] * desired_feedforward['right_s0']) + ',' + str(self._Kf['right_s1'] * desired_feedforward['right_s1']) + ',' + str(self._Kf['right_w0'] * desired_feedforward['right_w0']) + ',' + str(self._Kf['right_w1'] * desired_feedforward['right_w1']) + ',' + str(self._Kf['right_w2'] * desired_feedforward['right_w2']) + ',' + str(self._Kf['right_e0'] * desired_feedforward['right_e0']) + ',' + str(self._Kf['right_e1'] * desired_feedforward['right_e1']) + "\n")
        
        # ---> Also rosbag alongside trial <---
        f.close()

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
    global time_delay
    global rep_start
    global update_face
    global state
    global trial_cond
    global face_anim
    global phys_resp
    global stiff_case
    global freq

    # Start time
    start_time = time.time()
    rep_start = time.time()

    # Let user input experiment trial number (1-25, defined in dictionary below)
    parser = argparse.ArgumentParser(description = 'Input stimulus number.')
    parser.add_argument('integers', metavar = 'N', type = int, nargs = '+', help = 'an integer representing exp conds')
    foo = parser.parse_args()
    trial_cond = (foo.integers[0])

    # Access lookup table of conditions based on this input, assign appropriate values to global variables controlling trial conditions
    trial_stimuli = {1:[False,False,False,1.000,False], 2:[False,False,False,1.833,False], 3:[False,False,False,2.667,False],
                    4:[True,False,False,1.0000,False], 5:[True,False,False,1.833,False], 6:[True,False,False,2.667,False],
                    7:[False,True,False,1.000,False], 8:[False,True,False,1.833,False], 9:[False,True,False,2.667,False],
                    10:[False,False,True,1.000,False], 11:[False,False,True,1.833,False], 12:[False,False,True,2.667,False],
                    13:[True,True,False,1.000,False], 14:[True,True,False,1.833,False], 15:[True,True,False,2.667,False],
                    16:[False,True,True,1.000,False], 17:[False,True,True,1.833,False], 18:[False,True,True,2.667,False],
                    19:[True,False,True,1.000,False], 20:[True,False,True,1.833,False], 21:[True,False,True,2.667,False],
                    22:[True,True,True,1.000,False], 23:[True,True,True,1.833,False], 24:[True,True,True,2.667,False], 
                    25:[True,False,False,1.833,True]} # <---in last trial, ask for user's favorite set of conditions, use this
    
    # Break out selected dictionary entries into experiment trial variables
    face_anim = trial_stimuli[trial_cond][0] # set Boolean for turning face animation on or off
    phys_resp = trial_stimuli[trial_cond][1] # set Boolean for turning physical response on or off
    stiff_case = trial_stimuli[trial_cond][2] # set Boolean for turning stiffer stiffness on or off
    freq = trial_stimuli[trial_cond][3] # Hz, set robot clapping frequency
    trial_unlimited = trial_stimuli[trial_cond][4] # set Boolean identifying whether will be fixed length or extended free play

    # Intitialize High Five Arm object
    print("Initializing node... ")
    rospy.init_node("rsdk_joint_torque_springs_right")
    rospy.Subscriber ('/robot/accelerometer/right_accelerometer/state', Imu, callback)

    # Load face image on Baxter's screen
    send_image('/home/baxter/naomi_ws/src/test/src/BaxterHighFives/RestingFace.png')

    # Make high five arm object
    js = HighFiveArm('right')

    # Set control rate
    control_rate = rospy.Rate(1000)

    # Move robot to starting pose
    js._update_joint_angles()

    # For safety purposes, set the control rate command timeout.
    # If the specified number of command cycles are missed, the robot will timeout and disable.
    js._limb.set_command_timeout((1.0 / js._rate) * js._missed_cmds)

    # Calculate how many high five impacts should be expected in 20 sec based on the clapping frequency
    T = 1.000/freq
    num_claps = (20/T) + 1
    num_claps = int(num_claps)

    # Reinitialize start time to match desired trajectory
    time_delay = time.time() - start_time

    # Loop at specified rate commanding new joint torques
    while not rospy.is_shutdown():
        if not js._rs.state().enabled:
            rospy.logerr("Joint torque example failed to meet "
                         "specified control rate timeout.")
            break
        if face_anim:
            if state == 3 and update_face:
                # Set Baxter's expression to the ReactingFace
                send_image('/home/baxter/naomi_ws/src/test/src/BaxterHighFives/ReactingFace.png')
            if time.time()-rep_start > 0.2 and update_face:
                # Reset Baxter's expression to the original RestingFace
                send_image('/home/baxter/naomi_ws/src/test/src/BaxterHighFives/RestingFace.png')
                update_face = False
        if cback_loop_count <= num_claps or trial_unlimited:
            # Keep updating arm torques until desired number of hand-clapping cycles has been reached
            js._update_forces()
            control_rate.sleep()
        else:
            # Go back to start position once trial is done
            rospy.sleep(0.2)
            send_image('/home/baxter/naomi_ws/src/test/src/BaxterHighFives/RestingFace.png')
            js._update_end_pos()


if __name__ == "__main__":
    main()