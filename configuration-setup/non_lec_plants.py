
# Non learning enabled controller (LEC) based plants

import math
import numpy as np
from numpy import deg2rad
import matplotlib.pyplot as plt

import os.path
import sys
nxg_path = os.environ.get('NXG_PATH')
aerobench_path = nxg_path + 'AeroBenchVVPython/code/'
sys.path.append(aerobench_path)

from aerobench.run_f16_sim import run_f16_sim

from aerobench.visualize import plot

from aerobench.examples.gcas.gcas_autopilot import GcasAutopilot


class nonLECPlant(object):
    def __init__(self, model, steps=1000):
        self.model = model
        self.steps = steps
        self.mode = 0

    def get_controller_input(self, state):
        if self.model == 'CartPoleLinControl':
            M = 1
            g = 9.81

            sigma = state[0]
            theta = state[3]

            control = -1.1 * M * g * theta - sigma
            return control
        elif self.model == 'CartPoleLinControl2':
            control = -10.0 * state[0] + 289.83 * state[1] - 19.53 * state[2] + 63.25 * state[3]
            return control
        else:
            return 1

    def get_vel(self, state):
        vel = state[1]
        print("start mode " + str(self.mode))
        if self.mode == 0 and state[0] <= 0.01 and state[1] <= 0:
            print("setting mode 1")
            self.mode = 1
            vel = -0.75 * state[1]
        print(" mode is " + str(self.mode))

        return vel

    def get_mode(self, state):
        mode = 0
        if state[1] >= 0:
            mode = 1
        return mode

    def get_simulations(self, states, stepSize, steps):
        trajectories = []
        dim = len(states[0])
        if self.model == 'GCAS':
            for state in states:
                speed = []
                altitude = []
                throttle = []
                powers = []
                # vt = 540  # initial velocity (ft/sec)
                # power = 9  # engine power level (0-10)
                if dim == 2:
                    vt = state[0]
                    power = state[1]
                else:
                    vt = state[0]
                    power = 8.5

                # Default alpha & beta
                alpha = deg2rad(2.1215)  # Trim Angle of Attack (rad)
                beta = 0  # Side slip angle (rad)
                alt = 1000  # altitude (ft)
                phi = -math.pi / 8  # Roll angle from wings level (rad)
                theta = (-math.pi / 2) * 0.3  # Pitch angle from nose level (rad)
                psi = 0  # Yaw angle from North (rad)

                # Build Initial Condition Vectors
                init = [vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]
                # tmax = 3.51  # simulation time
                tmax = stepSize * steps

                ap = GcasAutopilot(init_mode='roll', stdout=True, gain_str='old')

                # step = 1 / 30
                step = stepSize
                res = run_f16_sim(init, tmax, ap, step=step, extended_states=True)

                # print(f"Simulation Completed in {round(res['runtime'], 3)} seconds")
                sim_states = res['states']
                inputs = res['u_list']
                assert len(sim_states) == len(inputs)
                traj = []
                for idx in range(len(sim_states)):
                    sim_state = sim_states[idx]
                    input = inputs[idx]
                    if dim == 2:
                        traj.append([sim_state[0], sim_state[12]])  # Vt and power
                    else:
                        traj.append(sim_state[0])
                    speed.append(sim_state[0])
                    altitude.append(sim_state[11])
                    throttle.append(input[0])
                    powers.append(sim_state[12])
                # fig = plt.figure()
                # plt.plot(speed)
                # plt.plot(powers)
                # plt.plot(altitude)
                # # plt.plot(throttle)
                # plt.show(fig)
                trajectories += [np.array(traj)]
        elif self.model == 'GCASInv':
            for state in states:
                speed = []
                altitude = []
                # vt = 540  # initial velocity (ft/sec)
                # power = 9  # engine power level (0-10)
                vt = state[0]
                power = state[1]

                # Default alpha & beta
                alpha = deg2rad(2.1215)  # Trim Angle of Attack (rad)
                beta = 0  # Side slip angle (rad)
                alt = 1000  # altitude (ft)
                phi = -math.pi * 0.9  # Roll angle from wings level (rad)
                theta = (-math.pi / 2) * 0.01  # Pitch angle from nose level (rad)
                psi = 0  # Yaw angle from North (rad)

                # Build Initial Condition Vectors
                init = [vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]
                # tmax = 3.51  # simulation time
                tmax = stepSize * steps

                ap = GcasAutopilot(init_mode='roll', stdout=True)

                # step = 1 / 30
                step = stepSize
                res = run_f16_sim(init, tmax, ap, step=step, extended_states=True, integrator_str='rk45')

                print(f"Simulation Completed in {round(res['runtime'], 3)} seconds")
                sim_states = res['states']
                traj = []
                fig = plt.figure()
                for sim_state in sim_states:
                    traj.append([sim_state[0], sim_state[12]])
                    speed.append(sim_state[0])
                    altitude.append(sim_state[11])
                plt.plot(speed)
                plt.plot(altitude)
                plt.show(fig)
                trajectories += [np.array(traj)]
        return trajectories