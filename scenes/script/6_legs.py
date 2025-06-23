from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import os
import random
import math
import numpy as np

# --- Global Variables (formerly RobotController attributes) ---
sim = None
simui = None
ui_handle = None # Global variable to store the UI handle

leg_num = 6  # Updated for 6 legs (R1, R2, R3, L1, L2, L3)
joint_num = 3
joint_compliance = True
springK = 8.0
springD = 0.2
k = 0.4
k_st = 0.4
MI = 0.05
ver_learn_time = 3.0
hor_learn_time = 7.0
test_start_time = 30
joint_state = 1

# Added X and Y lists for R3 and L3
leg_names = [f'R{i+1}' for i in range(leg_num // 2)] + [f'L{i+1}' for i in range(leg_num // 2)]

X = {leg: [] for leg in leg_names}
Y = {leg: [] for leg in leg_names}
# print("X and Y initialized for legs:", X)

# Added joint handles for R3 and L3
jointHandle_R1, jointHandle_R2, jointHandle_R3, jointHandle_L1, jointHandle_L2, jointHandle_L3 = {}, {}, {}, {}, {}, {}
# Added joint torques for R3 and L3
jointTorqueR1, jointTorqueR2, jointTorqueR3, jointTorqueL1, jointTorqueL2, jointTorqueL3 = {}, {}, {}, {}, {}, {}
# Added previous joint torques for R3 and L3
jointPrevTorqueR1, jointPrevTorqueR2, jointPrevTorqueR3, jointPrevTorqueL1, jointPrevTorqueL2, jointPrevTorqueL3 = {}, {}, {}, {}, {}, {}
# Added joint velocities for R3 and L3
jointVelR1, jointVelR2, jointVelR3, jointVelL1, jointVelL2, jointVelL3 = {}, {}, {}, {}, {}, {}
joint_Bias = [0, 0, 0]

# Added body markers for R3 and L3
# Generalized body handles for each leg
body_handles = {leg: None for leg in leg_names}
forceSensorHandle = None

# S02 CPG model
o1 = 0.01
o2 = 0.01
w11 = 1.4
w22 = 1.4
w12 = 0.18 + MI
w21 = -0.18 - MI

# lowpass
w_in = 0.1
w_rec = 1 - w_in

# stance gain control (added for R3 and L3)
e_st_gainR1 = [0, 0, 0]
e_st_gainR2 = [0, 0, 0]
e_st_gainR3 = [0, 0, 0]
e_st_gainL1 = [0, 0, 0]
e_st_gainL2 = [0, 0, 0]
e_st_gainL3 = [0, 0, 0]

e_sw_gainR1 = [0, 0, 0]
e_sw_gainR2 = [0, 0, 0]
e_sw_gainR3 = [0, 0, 0]
e_sw_gainL1 = [0, 0, 0]
e_sw_gainL2 = [0, 0, 0]
e_sw_gainL3 = [0, 0, 0]

e_sd_gainR1 = [0, 0, 0]
e_sd_gainR2 = [0, 0, 0]
e_sd_gainR3 = [0, 0, 0]
e_sd_gainL1 = [0, 0, 0]
e_sd_gainL2 = [0, 0, 0]
e_sd_gainL3 = [0, 0, 0]

e_st_gain = [0, 0, 0] # Keep these if used globally, otherwise remove
e_sw_gain = [0, 0, 0]
e_sd_gain = [0, 0, 0]

k_st = 0.4
k_sw = 0.4
initial_learning_time = 3.5
counter = 0
v_counter = 0
counter_step = 200
ver_learned = False
learned = False
rd_amp = 50
amplitude = 4
freq = 35
timelearned = 3.5
ep_length = 200
babbling_state = True
rebased_time = 50

k_x = 0.6
k_y = 0.0
k_z = 1.0

# Learned parameters hexa leg (pre-filled as in Lua) - Added for R3 and L3
e_st_gainR1[0] = 0.0020 # Python lists are 0-indexed
e_st_gainR2[0] = 0.0020
e_st_gainR3[0] = 0.0020 # Added
e_st_gainL1[0] = 0.0020
e_st_gainL2[0] = 0.0020
e_st_gainL3[0] = 0.0020 # Added

e_st_gainR1[1] = 0.4419
e_st_gainR2[1] = 0.4417
e_st_gainR3[1] = 0.4418 # Added (example value)
e_st_gainL1[1] = 0.4421
e_st_gainL2[1] = 0.4416
e_st_gainL3[1] = 0.4417 # Added (example value)

e_st_gainR1[2] = -0.8971
e_st_gainR2[2] = -0.8971
e_st_gainR3[2] = -0.8971 # Added (example value)
e_st_gainL1[2] = -0.8970
e_st_gainL2[2] = -0.8972
e_st_gainL3[2] = -0.8971 # Added (example value)

e_sw_gainR1[0] = -0.8003
e_sw_gainR2[0] = -0.7238
e_sw_gainR3[0] = -0.7600 # Added (example value)
e_sw_gainL1[0] = -0.7932
e_sw_gainL2[0] = -0.7545
e_sw_gainL3[0] = -0.7700 # Added (example value)

e_sw_gainR1[1] = 0.0230
e_sw_gainR2[1] = -0.2064
e_sw_gainR3[1] = -0.1000 # Added (example value)
e_sw_gainL1[1] = 0.0363
e_sw_gainL2[1] = -0.0480
e_sw_gainL3[1] = -0.0100 # Added (example value)

e_sw_gainR1[2] = -0.1766
e_sw_gainR2[2] = -0.0699
e_sw_gainR3[2] = -0.1200 # Added (example value)
e_sw_gainL1[2] = -0.1705
e_sw_gainL2[2] = 0.1975
e_sw_gainL3[2] = 0.0100 # Added (example value)

e_sd_gainR1[0] = -0.0942
e_sd_gainR2[0] = -0.0027
e_sd_gainR3[0] = -0.0500 # Added (example value)
e_sd_gainL1[0] = 0.0458
e_sd_gainL2[0] = -0.0422
e_sd_gainL3[0] = -0.0200 # Added (example value)

e_sd_gainR1[1] = 0.4358
e_sd_gainR2[1] = 0.4260
e_sd_gainR3[1] = 0.4300 # Added (example value)
e_sd_gainL1[1] = -0.4661
e_sd_gainL2[1] = -0.3937
e_sd_gainL3[1] = -0.4200 # Added (example value)

e_sd_gainR1[2] = 0.4700
e_sd_gainR2[2] = 0.5713
e_sd_gainR3[2] = 0.5200 # Added (example value)
e_sd_gainL1[2] = -0.4881
e_sd_gainL2[2] = -0.5641
e_sd_gainL3[2] = -0.5300 # Added (example value)


log_file = None # For CSV logging

# --- Helper Functions (formerly methods) ---
def _transposeMatrix(mat):
    return mat.T

def _multiplyMatrices(A, B):
    return A @ B

def _invertMatrix(matrix):
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        print("Matrix is singular and cannot be inverted.")
        return None

def _calculate_W(X, Y):
    X_T = _transposeMatrix(X)
    XT_X = _multiplyMatrices(X_T, X)
    XT_X_inv = _invertMatrix(XT_X)

    if XT_X_inv is None:
        return None

    XT_Y = _multiplyMatrices(X_T, Y)
    W = _multiplyMatrices(XT_X_inv, XT_Y)
    return W

def _normalized_weights_matrix(W):
    epsilon = 1e-8
    col_sums = np.sum(np.abs(W), axis=0) + epsilon
    W_normalized = W / col_sums
    return W_normalized

# --- UI Callback (now directly manipulates global variables) ---
def sliderChange(ui_handle_arg, _id, new_val):
    global k_x, k_y, k_z # Declare globals to modify them
    print(f"Slider changed: id={_id}, new_val={new_val}")
    if _id == 5001:
        k_x = new_val / 10.0
        simui.setLabelText(ui_handle_arg, 6001, f"k_x: {k_x:.1f}")
        print(f"k_x: {k_x:.1f}")
    elif _id == 5002:
        k_y = new_val / 10.0
        simui.setLabelText(ui_handle_arg, 6002, f"k_y: {k_y:.1f}")
        print(f"k_y: {k_y:.1f}")
    elif _id == 5003:
        k_z = new_val / 10.0
        simui.setLabelText(ui_handle_arg, 6003, f"k_z: {k_z:.1f}")
        print(f"k_z: {k_z:.1f}")

# --- Simulation Initialization ---
def init_simulation_objects():
    global sim, simui, ui_handle
    global bodyR1, bodyR2, bodyR3, bodyL1, bodyL2, bodyL3, forceSensorHandle
    global jointHandle_R1, jointHandle_R2, jointHandle_R3, jointHandle_L1, jointHandle_L2, jointHandle_L3
    global jointTorqueR1, jointTorqueR2, jointTorqueR3, jointTorqueL1, jointTorqueL2, jointTorqueL3
    global jointPrevTorqueR1, jointPrevTorqueR2, jointPrevTorqueR3, jointPrevTorqueL1, jointPrevTorqueL2, jointPrevTorqueL3
    global jointVelR1, jointVelR2, jointVelR3, jointVelL1, jointVelL2, jointVelL3

    print("Initializing simulation objects...")
    bodyR1 = sim.getObject(':/Body_markerR1')
    bodyR2 = sim.getObject(':/Body_markerR2')
    bodyR3 = sim.getObject(':/Body_markerR3') # Added
    bodyL1 = sim.getObject(':/Body_markerL1')
    bodyL2 = sim.getObject(':/Body_markerL2')
    bodyL3 = sim.getObject(':/Body_markerL3') # Added

    for i in range(joint_num):
        jointHandle_R1[i] = sim.getObject(f':/R1_J{i+1}')
        jointHandle_R2[i] = sim.getObject(f':/R2_J{i+1}')
        jointHandle_R3[i] = sim.getObject(f':/R3_J{i+1}') # Added
        jointHandle_L1[i] = sim.getObject(f':/L1_J{i+1}')
        jointHandle_L2[i] = sim.getObject(f':/L2_J{i+1}')
        jointHandle_L3[i] = sim.getObject(f':/L3_J{i+1}') # Added

        if joint_compliance:
            sim.setObjectInt32Param(jointHandle_R1[i], sim.jointintparam_dynctrlmode, sim.jointdynctrl_spring)
            sim.setObjectInt32Param(jointHandle_R2[i], sim.jointintparam_dynctrlmode, sim.jointdynctrl_spring)
            sim.setObjectInt32Param(jointHandle_R3[i], sim.jointintparam_dynctrlmode, sim.jointdynctrl_spring) # Added
            sim.setObjectInt32Param(jointHandle_L1[i], sim.jointintparam_dynctrlmode, sim.jointdynctrl_spring)
            sim.setObjectInt32Param(jointHandle_L2[i], sim.jointintparam_dynctrlmode, sim.jointdynctrl_spring)
            sim.setObjectInt32Param(jointHandle_L3[i], sim.jointintparam_dynctrlmode, sim.jointdynctrl_spring) # Added

            sim.setObjectFloatParam(jointHandle_R1[i], sim.jointfloatparam_kc_k, springK)
            sim.setObjectFloatParam(jointHandle_R2[i], sim.jointfloatparam_kc_k, springK)
            sim.setObjectFloatParam(jointHandle_R3[i], sim.jointfloatparam_kc_k, springK) # Added
            sim.setObjectFloatParam(jointHandle_L1[i], sim.jointfloatparam_kc_k, springK)
            sim.setObjectFloatParam(jointHandle_L2[i], sim.jointfloatparam_kc_k, springK)
            sim.setObjectFloatParam(jointHandle_L3[i], sim.jointfloatparam_kc_k, springK) # Added

            sim.setObjectFloatParam(jointHandle_R1[i], sim.jointfloatparam_kc_c, springD)
            sim.setObjectFloatParam(jointHandle_R2[i], sim.jointfloatparam_kc_c, springD)
            sim.setObjectFloatParam(jointHandle_R3[i], sim.jointfloatparam_kc_c, springD) # Added
            sim.setObjectFloatParam(jointHandle_L1[i], sim.jointfloatparam_kc_c, springD)
            sim.setObjectFloatParam(jointHandle_L2[i], sim.jointfloatparam_kc_c, springD)
            sim.setObjectFloatParam(jointHandle_L3[i], sim.jointfloatparam_kc_c, springD) # Added
        else:
            sim.setJointTargetForce(jointHandle_R1[i], 6.0)
            sim.setJointTargetForce(jointHandle_R2[i], 6.0)
            sim.setJointTargetForce(jointHandle_R3[i], 6.0) # Added
            sim.setJointTargetForce(jointHandle_L1[i], 6.0)
            sim.setJointTargetForce(jointHandle_L2[i], 6.0)
            sim.setJointTargetForce(jointHandle_L3[i], 6.0) # Added

        jointTorqueR1[i] = 0.0
        jointTorqueR2[i] = 0.0
        jointTorqueR3[i] = 0.0 # Added
        jointTorqueL1[i] = 0.0
        jointTorqueL2[i] = 0.0
        jointTorqueL3[i] = 0.0 # Added

        jointPrevTorqueR1[i] = 0.0
        jointPrevTorqueR2[i] = 0.0
        jointPrevTorqueR3[i] = 0.0 # Added
        jointPrevTorqueL1[i] = 0.0
        jointPrevTorqueL2[i] = 0.0
        jointPrevTorqueL3[i] = 0.0 # Added

        jointVelR1[i] = 0.0
        jointVelR2[i] = 0.0
        jointVelR3[i] = 0.0 # Added
        jointVelL1[i] = 0.0
        jointVelL2[i] = 0.0
        jointVelL3[i] = 0.0 # Added

    forceSensorHandle = sim.getObject(':/forceSensor')
    print("forceSensorHandle: ", forceSensorHandle)

    xml = f"""
        <ui title="Tuning Parameters" closeable="true" position="50,50" size="300,200">
            <hslider id="5001" tick-position="above" tick-interval="20" minimum="-10" maximum="10" on-change="sliderChange" value="{int(k_x * 10)}"/>
            <label id="6001" text="k_x: {k_x:.1f}"/>

            <hslider id="5002" tick-position="above" tick-interval="20" minimum="-10" maximum="10" on-change="sliderChange" value="{int(k_y * 10)}"/>
            <label id="6002" text="k_y: {k_y:.1f}" />

            <hslider id="5003" tick-position="above" tick-interval="20" minimum="-10" maximum="10" on-change="sliderChange" value="{int(k_z * 10)}"/>
            <label  id="6003" text="k_z: {k_z:.1f}"/>
        </ui>
    """
    ui_handle = simui.create(xml)
    simui.setPosition(ui_handle, 0, 0)
    print("UI created and positioned.")

# --- Actuation Step ---
def actuation_step():
    global o1, o2
    o1 = math.tanh(o1 * w11 + o2 * w12)
    o2 = math.tanh(o2 * w22 + o1 * w21)

    ps = 1 if o1 > 0 else 0

    if learned:
        for i in range(joint_num):
            if ps == 1:  # Swing phase
                sim.setJointTargetPosition(jointHandle_R1[i], joint_Bias[i] + o1 * e_st_gainR1[i] * k_z - o1 * e_sw_gainR1[i] * k_x - o1 * e_sd_gainR1[i] * k_y)
                sim.setJointTargetPosition(jointHandle_R2[i], joint_Bias[i] - o1 * e_st_gainR2[i] * k_z * 0 + o1 * e_sw_gainR2[i] * k_x + o1 * e_sd_gainR2[i] * k_y)
                sim.setJointTargetPosition(jointHandle_R3[i], joint_Bias[i] + o1 * e_st_gainR3[i] * k_z - o1 * e_sw_gainR3[i] * k_x - o1 * e_sd_gainR3[i] * k_y) # Added
                sim.setJointTargetPosition(jointHandle_L1[i], joint_Bias[i] - o1 * e_st_gainL1[i] * k_z * 0 + o1 * e_sw_gainL1[i] * k_x + o1 * e_sd_gainL1[i] * k_y)
                sim.setJointTargetPosition(jointHandle_L2[i], joint_Bias[i] + o1 * e_st_gainL2[i] * k_z - o1 * e_sw_gainL2[i] * k_x - o1 * e_sd_gainL2[i] * k_y)
                sim.setJointTargetPosition(jointHandle_L3[i], joint_Bias[i] - o1 * e_st_gainL3[i] * k_z * 0 + o1 * e_sw_gainL3[i] * k_x + o1 * e_sd_gainL3[i] * k_y) # Added
            elif ps == 0:  # Stance phase
                sim.setJointTargetPosition(jointHandle_R1[i], joint_Bias[i] + o1 * e_st_gainR1[i] * k_z * 0 - o1 * e_sw_gainR1[i] * k_x - o1 * e_sd_gainR1[i] * k_y)
                sim.setJointTargetPosition(jointHandle_R2[i], joint_Bias[i] - o1 * e_st_gainR2[i] * k_z + o1 * e_sw_gainR2[i] * k_x + o1 * e_sd_gainR2[i] * k_y)
                sim.setJointTargetPosition(jointHandle_R3[i], joint_Bias[i] + o1 * e_st_gainR3[i] * k_z * 0 - o1 * e_sw_gainR3[i] * k_x - o1 * e_sd_gainR3[i] * k_y) # Added
                sim.setJointTargetPosition(jointHandle_L1[i], joint_Bias[i] - o1 * e_st_gainL1[i] * k_z + o1 * e_sw_gainL1[i] * k_x + o1 * e_sd_gainL1[i] * k_y)
                sim.setJointTargetPosition(jointHandle_L2[i], joint_Bias[i] + o1 * e_st_gainL2[i] * k_z * 0 - o1 * e_sw_gainL2[i] * k_x - o1 * e_sd_gainL2[i] * k_y)
                sim.setJointTargetPosition(jointHandle_L3[i], joint_Bias[i] - o1 * e_st_gainL3[i] * k_z + o1 * e_sw_gainL3[i] * k_x + o1 * e_sd_gainL3[i] * k_y) # Added

# --- Sensing Step and Learning Logic ---
def sensing_step():
    global ver_learned, v_counter, learned, counter, babbling_state, joint_state
    global e_st_gainR1, e_st_gainR2, e_st_gainR3, e_st_gainL1, e_st_gainL2, e_st_gainL3 # Added R3, L3
    global e_sw_gainR1, e_sw_gainR2, e_sw_gainR3, e_sw_gainL1, e_sw_gainL2, e_sw_gainL3 # Added R3, L3
    global e_sd_gainR1, e_sd_gainR2, e_sd_gainR3, e_sd_gainL1, e_sd_gainL2, e_sd_gainL3 # Added R3, L3

    for i in range(joint_num):
        jointTorqueR1[i] = sim.getJointForce(jointHandle_R1[i])
        jointTorqueR2[i] = sim.getJointForce(jointHandle_R2[i])
        jointTorqueR3[i] = sim.getJointForce(jointHandle_R3[i]) # Added
        jointTorqueL1[i] = sim.getJointForce(jointHandle_L1[i])
        jointTorqueL2[i] = sim.getJointForce(jointHandle_L2[i])
        jointTorqueL3[i] = sim.getJointForce(jointHandle_L3[i]) # Added

        jointVelR1[i] = sim.getJointVelocity(jointHandle_R1[i])
        jointVelR2[i] = sim.getJointVelocity(jointHandle_R2[i])
        jointVelR3[i] = sim.getJointVelocity(jointHandle_R3[i]) # Added
        jointVelL1[i] = sim.getJointVelocity(jointHandle_L1[i])
        jointVelL2[i] = sim.getJointVelocity(jointHandle_L2[i])
        jointVelL3[i] = sim.getJointVelocity(jointHandle_L3[i]) # Added

    linearVelocityR1, angularVelocityR1 = sim.getObjectVelocity(bodyR1)
    linearVelocityR2, angularVelocityR2 = sim.getObjectVelocity(bodyR2)
    linearVelocityR3, angularVelocityR3 = sim.getObjectVelocity(bodyR3) # Added
    linearVelocityL1, angularVelocityL1 = sim.getObjectVelocity(bodyL1)
    linearVelocityL2, angularVelocityL2 = sim.getObjectVelocity(bodyL2)
    linearVelocityL3, angularVelocityL3 = sim.getObjectVelocity(bodyL3) # Added

    tot_torqueR1 = sum(abs(t) for t in jointTorqueR1.values())
    tot_torqueR2 = sum(abs(t) for t in jointTorqueR2.values())
    tot_torqueR3 = sum(abs(t) for t in jointTorqueR3.values()) # Added
    tot_torqueL1 = sum(abs(t) for t in jointTorqueL1.values())
    tot_torqueL2 = sum(abs(t) for t in jointTorqueL2.values())
    tot_torqueL3 = sum(abs(t) for t in jointTorqueL3.values()) # Added

    if not ver_learned:
        for i in range(joint_num):
            e_st_gainR1[i] = jointTorqueR1[i] / tot_torqueR1 if tot_torqueR1 != 0 else 0
            e_st_gainR2[i] = jointTorqueR2[i] / tot_torqueR2 if tot_torqueR2 != 0 else 0
            e_st_gainR3[i] = jointTorqueR3[i] / tot_torqueR3 if tot_torqueR3 != 0 else 0 # Added
            e_st_gainL1[i] = jointTorqueL1[i] / tot_torqueL1 if tot_torqueL1 != 0 else 0
            e_st_gainL2[i] = jointTorqueL2[i] / tot_torqueL2 if tot_torqueL2 != 0 else 0
            e_st_gainL3[i] = jointTorqueL3[i] / tot_torqueL3 if tot_torqueL3 != 0 else 0 # Added

        v_counter += 1
        if v_counter > 100:
            ver_learned = True

    timeStep = sim.getSimulationTime()
    if not learned and ver_learned and timeStep > 1:
        if babbling_state:
            if counter % freq == 0:
                # This random force application now affects R1, but you might want to generalize this
                # to cycle through all legs, or apply different random forces for different learning phases.
                sim.setJointTargetPosition(jointHandle_R1[joint_state - 1], (random.uniform(0,2) - 1) / 10 * amplitude)
                print("Joint State: ", joint_state)

            X['R1'].append([jointVelR1[0], jointVelR1[1], jointVelR1[2]])
            Y['R1'].append([linearVelocityR1[0], linearVelocityR1[1], linearVelocityR1[2]])
            X['R2'].append([jointVelR2[0], jointVelR2[1], jointVelR2[2]])
            Y['R2'].append([linearVelocityR2[0], linearVelocityR2[1], linearVelocityR2[2]])
            X['R3'].append([jointVelR3[0], jointVelR3[1], jointVelR3[2]])
            Y['R3'].append([linearVelocityR3[0], linearVelocityR3[1], linearVelocityR3[2]])
            X['L1'].append([jointVelL1[0], jointVelL1[1], jointVelL1[2]])
            Y['L1'].append([linearVelocityL1[0], linearVelocityL1[1], linearVelocityL1[2]])
            X['L2'].append([jointVelL2[0], jointVelL2[1], jointVelL2[2]])
            Y['L2'].append([linearVelocityL2[0], linearVelocityL2[1], linearVelocityL2[2]])
            X['L3'].append([jointVelL3[0], jointVelL3[1], jointVelL3[2]])
            Y['L3'].append([linearVelocityL3[0], linearVelocityL3[1], linearVelocityL3[2]])
            counter += 1
            if counter > ep_length and babbling_state:
                babbling_state = False
                counter = 0
                joint_state += 1
        else:
            sim.setJointTargetPosition(jointHandle_R1[joint_state - 2], 0.0) # This needs to be generalized
            if counter > rebased_time:
                babbling_state = True
                counter = 0
            counter += 1
            # The learning trigger condition now checks against leg_num
            if joint_state > joint_num and counter > rebased_time - 2:
                learned = True
                print(f"Learned: {learned}")
                for i in range(joint_num):
                    sim.setJointTargetPosition(jointHandle_R1[i], 0)
                    sim.setJointTargetPosition(jointHandle_R2[i], 0)
                    sim.setJointTargetPosition(jointHandle_R3[i], 0) # Added
                    sim.setJointTargetPosition(jointHandle_L1[i], 0)
                    sim.setJointTargetPosition(jointHandle_L2[i], 0)
                    sim.setJointTargetPosition(jointHandle_L3[i], 0) # Added

                W_R1 = _calculate_W(np.array(X['R1']), np.array(Y['R1']))
                W_R2 = _calculate_W(np.array(X['R2']), np.array(Y['R2']))
                W_R3 = _calculate_W(np.array(X['R3']), np.array(Y['R3']))
                W_L1 = _calculate_W(np.array(X['L1']), np.array(Y['L1']))
                W_L2 = _calculate_W(np.array(X['L2']), np.array(Y['L2']))
                W_L3 = _calculate_W(np.array(X['L3']), np.array(Y['L3']))

                W_R1_norm = _normalized_weights_matrix(W_R1)
                W_R2_norm = _normalized_weights_matrix(W_R2)
                W_R3_norm = _normalized_weights_matrix(W_R3) # Added
                W_L1_norm = _normalized_weights_matrix(W_L1)
                W_L2_norm = _normalized_weights_matrix(W_L2)
                W_L3_norm = _normalized_weights_matrix(W_L3) # Added

                print("--- Normalized Weights ---")
                print('W_R1_norm:\n', W_R1_norm)
                print('W_R2_norm:\n', W_R2_norm)
                print('W_R3_norm:\n', W_R3_norm) # Added
                print('W_L1_norm:\n', W_L1_norm)
                print('W_L2_norm:\n', W_L2_norm)
                print('W_L3_norm:\n', W_L3_norm) # Added

                for i in range(joint_num):
                    e_sw_gainR1[i] = W_R1_norm[i, 0]
                    e_sw_gainR2[i] = W_R2_norm[i, 0]
                    e_sw_gainR3[i] = W_R3_norm[i, 0] # Added
                    e_sw_gainL1[i] = W_L1_norm[i, 0]
                    e_sw_gainL2[i] = W_L2_norm[i, 0]
                    e_sw_gainL3[i] = W_L3_norm[i, 0] # Added

                    e_sd_gainR1[i] = W_R1_norm[i, 1]
                    e_sd_gainR2[i] = W_R2_norm[i, 1]
                    e_sd_gainR3[i] = W_R3_norm[i, 1] # Added
                    e_sd_gainL1[i] = W_L1_norm[i, 1]
                    e_sd_gainL2[i] = W_L2_norm[i, 1]
                    e_sd_gainL3[i] = W_L3_norm[i, 1] # Added

                for i in range(joint_num):
                    print(f"e_st_gainR1[{i}] = {e_st_gainR1[i]:.4f}")
                    print(f"e_st_gainR2[{i}] = {e_st_gainR2[i]:.4f}")
                    print(f"e_st_gainR3[{i}] = {e_st_gainR3[i]:.4f}") # Added
                    print(f"e_st_gainL1[{i}] = {e_st_gainL1[i]:.4f}")
                    print(f"e_st_gainL2[{i}] = {e_st_gainL2[i]:.4f}")
                    print(f"e_st_gainL3[{i}] = {e_st_gainL3[i]:.4f}") # Added

                for i in range(joint_num):
                    print(f"e_sw_gainR1[{i}] = {e_sw_gainR1[i]:.4f}")
                    print(f"e_sw_gainR2[{i}] = {e_sw_gainR2[i]:.4f}")
                    print(f"e_sw_gainR3[{i}] = {e_sw_gainR3[i]:.4f}") # Added
                    print(f"e_sw_gainL1[{i}] = {e_sw_gainL1[i]:.4f}")
                    print(f"e_sw_gainL2[{i}] = {e_sw_gainL2[i]:.4f}")
                    print(f"e_sw_gainL3[{i}] = {e_sw_gainL3[i]:.4f}") # Added

                for i in range(joint_num):
                    print(f"e_sd_gainR1[{i}] = {e_sd_gainR1[i]:.4f}")
                    print(f"e_sd_gainR2[{i}] = {e_sd_gainR2[i]:.4f}")
                    print(f"e_sd_gainR3[{i}] = {e_sd_gainR3[i]:.4f}") # Added
                    print(f"e_sd_gainL1[{i}] = {e_sd_gainL1[i]:.4f}")
                    print(f"e_sd_gainL2[{i}] = {e_sd_gainL2[i]:.4f}")
                    print(f"e_sd_gainL3[{i}] = {e_sd_gainL3[i]:.4f}") # Added

# --- Cleanup Function ---
def cleanup():
    global simui, ui_handle, log_file
    print("Cleaning up...")
    if simui and ui_handle:
        simui.destroy(ui_handle)
    if log_file:
        log_file.close()

# --- Main Program Logic ---
def main():
    global sim, simui, ui_handle # Declare globals to assign to them
    print('---------- Program started ----------')
    # Make sure you have a CoppeliaSim scene with R3 and L3 legs defined,
    # with corresponding joint and body marker names like 'R3_J1', 'Body_markerR3', etc.
    scene_file_path = os.path.abspath("/home/binggwong/git/SOMAL/scenes/archived/SOLMbot_hex_6legs_clean.ttt") # Changed to 6-leg scene
    # scene_file_path = os.path.abspath("/home/binggwong/git/SOMAL/scenes/archived/SOLMbot_hex_4l_clean.ttt")

    client = RemoteAPIClient()
    sim = client.require('sim')
    simui = client.require('simUI')
    sim.addLog(sim.verbosity_scriptinfos, '---------- Simulation started ----------')

    try:
        print(f"Loading scene: {scene_file_path}")
        sim.loadScene(scene_file_path)
    except Exception as e:
        print(f"Failed to load scene: {e}")
        return

    init_simulation_objects() # Initialize handles and UI

    sim.setStepping(True)
    sim.startSimulation()

    try:
        max_sim_time = 9999
        while (t := sim.getSimulationTime()) < max_sim_time:
            actuation_step()
            sensing_step()

            if int(t * 10) % 10 == 0:
                pass # print(f'Simulation time: {t:.2f} [s]')

            sim.step()

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    finally:
        sim.stopSimulation()
        cleanup()
        print('---------- Program ended ----------')
        sim.addLog(sim.verbosity_scriptinfos, '---------- Simulation ended ----------')

if __name__ == "__main__":
    main()