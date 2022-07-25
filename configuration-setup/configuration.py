import numpy as np

from sampler import generateRandomStates, generateSuperpositionSampler
from ODESolver import generateTrajectories, plotTrajectories
from lec_disc_dyn_plants import DnnController, Plant
from non_lec_plants import nonLECPlant
import json
import random
from os import path
from frechet import norm
import os.path
# from circleRandom import generate_points_in_circle

nxg_path = os.environ.get('NXG_PATH')
control_dir = nxg_path + 'controllers-yml-files/'


class configuration(object):
    def __init__(self, stepSize=0.01, steps=100, dynamics='None', dimensions=2, lowerBound=[],
                upperBound=[], gradient_run=False, disc=False):
        self.stepSize = stepSize
        self.steps = steps
        self.dynamics = dynamics
        self.dimensions = dimensions
        self.lowerBoundArray = lowerBound
        self.upperBoundArray = upperBound
        self.trajectories = []
        self.states = []
        self.neighbors = 10
        self.grad_run = gradient_run
        self.discrete_dyn = disc
        self.samples = 10

    def setTrajectories(self, trajectories):
        self.trajectories = trajectories

    def getTrajectories(self):
        return self.trajectories

    def setDynamics(self, dynamics):
        self.dynamics = dynamics

    def setLowerBound(self, lowerBound):
        self.lowerBoundArray = lowerBound

    def setUpperBound(self, upperBound):
        self.upperBoundArray = upperBound

    def setSamples(self, samples):
        self.samples = samples

    def setSteps(self, steps):
        self.steps = steps

    def setStepSize(self, stepSize):
        self.stepSize = stepSize

    def getStepSize(self):
        return self.stepSize

    def getDynamics(self):
        return self.dynamics

    def setDiscrete(self):
        self.discrete_dyn = True

    # Only for gradient run
    def setNeighbors(self, neighbors):
        assert self.grad_run is True
        self.neighbors = neighbors

    def getGradientRun(self):
        return self.grad_run

    def setGradientRun(self, grad_run):
        self.grad_run = grad_run

    def load_i_states_4m_file(self):
        r_states_fname = nxg_path + '/eval-emsoft/training-samples/' + self.dynamics + '_r_states.txt'
        r_states_f = open(r_states_fname, 'r')
        states = json.loads(r_states_f.read())
        r_states = []
        for idx in range(self.samples):
            r_states.append(states[idx])
        print(len(r_states))
        return r_states

    def generate_points_in_circle(self, n_samples=100, dim=2):
        num_samples = n_samples
        r_scale = 1

        points = []
        if dim == 2:
            # make a simple unit circle
            theta = np.linspace(0, 2 * np.pi, num_samples)
            # a, b = r_scale * np.cos(theta), r_scale * np.sin(theta)

            # generate the points
            # theta = np.random.rand((num_samples)) * (2 * np.pi)
            # r = r_scale * np.random.rand((num_samples))
            r = r_scale
            x, y = r * np.cos(theta), r * np.sin(theta)

            for idx in range(len(x)):
                point = [x[idx], y[idx]]
                points.append(point)
        elif dim == 3:
            for idx in range(n_samples):
                u = np.random.normal(0, 1)
                v = np.random.normal(0, 1)
                w = np.random.normal(0, 1)
                r = np.random.rand(1) ** (1. / 3)
                norm = (u * u + v * v + w * w) ** (0.5)
                (x, y, z) = r * (u, v, w) / norm
                # print("radius" + str((x**2 + y**2 + z**2) * 0.5))
                point = [x, y, z]
                points.append(point)
            # print(points)
        else:
            for idx in range(n_samples):
                u = np.random.normal(0, 1, dim)  # an array of d normally distributed random variables
                norm = np.sum(u ** 2) ** (0.5)
                r = np.random.rand(1) ** (1.0 / dim)
                x = r * u / norm
                point = []
                for d in range(dim):
                    point.append(x[d])
                points.append(point)
        return points

    def generateTrajectories(self, scaling=0.01, samples=None, r_states=None, dump_i_states=False, sims_f=None):
        if samples is not None:
            self.samples = samples
        plant = None
        dnn_cntrl_fname = None
        dnn_transform_obj = None
        if self.dynamics is 'MountainCarDisc':
            dnn_cntrl_fname = control_dir + 'verisig/mountain_car/' + 'sig16x16.yml'
        if self.dynamics is 'QuadrotorDisc':
            dnn_cntrl_fname = control_dir + 'verisig/quadrotor/' + 'tanh20x20.yml'
        if self.dynamics is 'InvPendulumDisc':
            dnn_cntrl_fname = control_dir + 'ARCH-2019/Inverted_pendulum/' + 'IPController.yml'
        if self.dynamics is 'OtherBench1Disc':
            dnn_cntrl_fname = control_dir + 'ARCH-2019/Benchmark1/' + 'controller.yml'
        if self.dynamics is 'ABSDisc':
            dnn_cntrl_fname = control_dir + 'NNV/ABS/' + 'controller.yml'
            dnn_tf_fname = control_dir + 'NNV/ABS/' + 'transform.yml'
            dnn_transform_obj = DnnController(dnn_tf_fname, 2)
        if self.dynamics is 'MountainCarCont':  # don't have a controller file for this
            dnn_cntrl_fname = control_dir + 'verisig/mountain_car/' + 'sig16x16.yml'
        if self.dynamics is 'OtherBenchC1':
            dnn_cntrl_fname = control_dir + 'ARCH-2019/Benchmark1/' + 'controller.yml'
        if self.dynamics is 'OtherBenchC2':
            dnn_cntrl_fname = control_dir + 'ARCH-2019/Benchmark2/' + 'controller.yml'
        if self.dynamics is 'OtherBenchC3':
            dnn_cntrl_fname = control_dir + 'ARCH-2019/Benchmark3/' + 'controller.yml'
        if self.dynamics is 'OtherBenchC4':
            dnn_cntrl_fname = control_dir + 'ARCH-2019/Benchmark4/' + 'controller.yml'
        if self.dynamics is 'OtherBenchC5':
            dnn_cntrl_fname = control_dir + 'ARCH-2019/Benchmark5/' + 'controller.yml'
        if self.dynamics is 'OtherBenchC6':
            dnn_cntrl_fname = control_dir + 'ARCH-2019//Benchmark6/' + 'controller.yml'
        if self.dynamics is 'OtherBenchC7':
            dnn_cntrl_fname = control_dir + 'ARCH-2019/Benchmark7/' + 'controller.yml'
        if self.dynamics is 'OtherBenchC8':
            dnn_cntrl_fname = control_dir + 'ARCH-2019/Benchmark8/' + 'controller.yml'
        if self.dynamics is 'ACCNonLinear3L':
            dnn_cntrl_fname = control_dir + 'ARCH-2019/ACC/' + 'acc_controller_3_20.yml'
        if self.dynamics is 'ACCNonLinear5L':
            dnn_cntrl_fname = control_dir + 'ARCH-2019/ACC/' + 'acc_controller_5_20.yml'
        if self.dynamics is 'ACCNonLinear7L':
            dnn_cntrl_fname = control_dir + 'ARCH-2019/ACC/' + 'acc_controller_7_20.yml'
        if self.dynamics is 'ACCNonLinear10L':
            dnn_cntrl_fname = control_dir + 'ARCH-2019/ACC/' + 'acc_controller_10_20.yml'
        if self.dynamics is 'OtherBenchC9':
            dnn_cntrl_fname = control_dir + 'ARCH-2020/Benchmark9-TORA/' + 'controllerTora_nnv.yml'
        if self.dynamics is 'OtherBenchC9Tanh':
            dnn_cntrl_fname = control_dir + 'ARCH-2020/Benchmark9-TORA-Tanh/' + 'controllerTora_nnv_tanh.yml'
        if self.dynamics is 'OtherBenchC9Sigmoid':
            dnn_cntrl_fname = control_dir + 'ARCH-2020/Benchmark9-TORA-Sigmoid/' + 'controllerTora_nnv_sigmoid.yml'
        if self.dynamics is 'SinglePendulum':
            dnn_cntrl_fname = control_dir + 'ARCH-2020/SinglePendulum/' + 'controller_single_pendulum.yml'
        if self.dynamics is 'DoublePendulumLess':
            dnn_cntrl_fname = control_dir + 'ARCH-2020/DoublePendulum/' + 'controller_double_pendulum_less.yml'
        if self.dynamics is 'DoublePendulumMore':
            dnn_cntrl_fname = control_dir + 'ARCH-2020/DoublePendulum/' + 'controller_double_pendulum_more.yml'
        if self.dynamics is 'OtherBenchC10':
            dnn_cntrl_fname = control_dir + 'ARCH-2020/Benchmark10-Unicycle/' + 'controller10_unicycle.yml'
        if self.dynamics is 'InvPendulumC':
            dnn_cntrl_fname = control_dir + 'ARCH-2019/Inverted-pendulum/' + 'IPController.yml'
        if self.dynamics is 'Airplane':
            dnn_cntrl_fname = control_dir + 'ARCH-2020/Airplane/' + 'controller_airplane.yml'
        if self.dynamics is 'CartPole':
            dnn_cntrl_fname = control_dir + 'ARCH-2019/Cart-pole/' + 'Cartpole_controller.yml'
        if self.dynamics is 'CartPoleTanh':
            dnn_cntrl_fname = control_dir + 'ARCH-2019/Cart-pole/' + 'Cartpole_controller_tanh.yml'
        if self.dynamics is 'VertCAS':
            dnn_cntrl_fname = control_dir + 'ARCH-2020/VCAS/' + 'VertCAS_1.yml'
            dnn_controller_obj = DnnController(dnn_cntrl_fname, self.dimensions)
            plant = Plant('VertCAS', dnn_controller_obj, None, self.steps)
            dnn_cntrl_fname = control_dir + 'ARCH-2020/VCAS/' + 'VertCAS_2.yml'
            dnn_controller_obj2 = DnnController(dnn_cntrl_fname, self.dimensions)
            plant.append_controller(dnn_controller_obj2)
            dnn_cntrl_fname = control_dir + 'ARCH-2020/VCAS/' + 'VertCAS_3.yml'
            dnn_controller_obj3 = DnnController(dnn_cntrl_fname, self.dimensions)
            plant.append_controller(dnn_controller_obj3)
            dnn_cntrl_fname = control_dir + 'ARCH-2020/VCAS/' + 'VertCAS_4.yml'
            dnn_controller_obj4 = DnnController(dnn_cntrl_fname, self.dimensions)
            plant.append_controller(dnn_controller_obj4)
            dnn_cntrl_fname = control_dir + 'ARCH-2020/VCAS/' + 'VertCAS_5.yml'
            dnn_controller_obj5 = DnnController(dnn_cntrl_fname, self.dimensions)
            plant.append_controller(dnn_controller_obj5)
            dnn_cntrl_fname = control_dir + 'ARCH-2020/VCAS/' + 'VertCAS_6.yml'
            dnn_controller_obj6 = DnnController(dnn_cntrl_fname, self.dimensions)
            plant.append_controller(dnn_controller_obj6)
            dnn_cntrl_fname = control_dir + 'ARCH-2020/VCAS/' + 'VertCAS_7.yml'
            dnn_controller_obj7 = DnnController(dnn_cntrl_fname, self.dimensions)
            plant.append_controller(dnn_controller_obj7)
            dnn_cntrl_fname = control_dir + 'ARCH-2020/VCAS/' + 'VertCAS_8.yml'
            dnn_controller_obj8 = DnnController(dnn_cntrl_fname, self.dimensions)
            plant.append_controller(dnn_controller_obj8)
            dnn_cntrl_fname = control_dir + 'ARCH-2020/VCAS/' + 'VertCAS_9.yml'
            dnn_controller_obj9 = DnnController(dnn_cntrl_fname, self.dimensions)
            plant.append_controller(dnn_controller_obj9)

        if dnn_cntrl_fname is not None and plant is None:
            if self.discrete_dyn is True:
                dnn_controller_obj = DnnController(dnn_cntrl_fname, self.dimensions)
                plant = Plant(self.dynamics, dnn_controller_obj, dnn_transform_obj, self.steps)
            elif self.discrete_dyn is False:
                dnn_controller_obj = DnnController(dnn_cntrl_fname, self.dimensions)
                plant = Plant(self.dynamics, dnn_controller_obj, dnn_transform_obj, self.steps)
                plant.dnn_controllers[0].parseDNNYML(self.dynamics)
        elif self.dynamics is 'CartPoleLinControl':
            plant = nonLECPlant('CartPoleLinControl')
        elif self.dynamics is 'CartPoleLinControl2':
            plant = nonLECPlant('CartPoleLinControl2')
        elif self.dynamics is 'BouncingBall':
            plant = nonLECPlant('BouncingBall')
        elif self.dynamics is 'GCAS' or self.dynamics is 'GCASInv':
            plant = nonLECPlant(self.dynamics)

        if r_states is None:
            r_states = generateRandomStates(self.samples, self.lowerBoundArray, self.upperBoundArray)
        if dump_i_states is True:
            r_states_fname = nxg_path + '/eval-emsoft/training-samples/' + self.dynamics + '_r_states.txt'
            if path.exists(r_states_fname):
                os.remove(r_states_fname)
            r_states_f = open(r_states_fname, 'w')
            r_states_f.write(json.dumps(r_states))
            r_states_f.close()

        if plant is not None and self.discrete_dyn is True:
            r_trajectories = plant.getSimulations(states=r_states)
        elif plant is not None and (self.dynamics == 'GCAS' or self.dynamics == 'GCASInv'):
            r_trajectories = plant.get_simulations(states=r_states, stepSize=self.stepSize, steps=self.steps)
        else:
            r_time = np.linspace(0, self.stepSize * self.steps, self.steps + 1)
            r_trajectories = generateTrajectories(self.dynamics, r_states, r_time, plant)

        if self.grad_run is True:
            vecs_in_unit_circle = self.generate_points_in_circle(n_samples=self.neighbors, dim=self.dimensions)

            # print(vecs_in_unit_circle)
            delta_vecs = []
            for vec in vecs_in_unit_circle:
                delta_vec = [val*scaling for val in vec]
                delta_vecs.append(delta_vec)
            # print(delta_vecs)
            trajectories = []
            for idx in range(len(r_states)):
                trajectories.append(r_trajectories[idx])
                # print(r_trajectories[idx])
                r_state = r_states[idx]
                for delta_vec in delta_vecs:
                    # print(r_state, delta_vec)
                    neighbor_state = [r_state[i] + delta_vec[i] for i in range(len(delta_vec))]
                    if plant is not None and self.discrete_dyn is True:
                        neighbor_trajectory = plant.getSimulations(states=[neighbor_state], do_not_parse=True)[0]
                    elif plant is not None and (self.dynamics == 'GCAS' or self.dynamics == 'GCASInv'):
                        neighbor_trajectory = plant.get_simulations(states=[neighbor_state], stepSize=self.stepSize,
                                                                    steps=self.steps)[0]
                    else:
                        # Added below r_time statement to take care of some warning - haven't tested it though.
                        # It has been working without this. If it causes an issue, comment it.
                        r_time = np.linspace(0, self.stepSize * self.steps, self.steps + 1)
                        neighbor_trajectory = generateTrajectories(self.dynamics, [neighbor_state], r_time, plant)[0]
                    trajectories.append(neighbor_trajectory)
            self.trajectories = trajectories
        else:
            self.trajectories = r_trajectories

        if sims_f is not None:
            for traj in self.trajectories:
                sims_f.write(str(traj))
                sims_f.write('\n')
        return self.trajectories

    def showTrajectories(self, trajectories=None, xindex=0, yindex=1, dimwise=False):
        if trajectories is None:
            plotTrajectories(self.trajectories, xindex=xindex, yindex=yindex, dimwise=dimwise)
        else:
            plotTrajectories(trajectories, xindex=xindex, yindex=yindex, dimwise=dimwise)

    def storeStatesRandomSample(self):
        self.states = generateRandomStates(self.samples, self.lowerBoundArray, self.upperBoundArray)

    def storeTrajectories(self):

        if self.states == [] :
            self.storeStatesRandomSample()

        assert not (self.states == [])
        # assert self.lowerBoundArray is not [] and self.upperBoundArray is not []
        self.time = np.linspace(0, self.stepSize * self.steps, self.steps + 1)
        self.trajectories = generateTrajectories(self.dynamics, self.states, self.time)

    def dumpTrajectoriesStates(self, eval_var):
        f_name = "../models_trajs/states_"
        f_name = f_name + eval_var + "_"
        f_name = f_name + self.dynamics
        f_name = f_name + ".txt"
        if path.exists(f_name):
            os.remove(f_name)
        states_f = open(f_name, "w")
        for idx in range(len(self.states)):
            states_f.write(str(self.states[idx]))
            states_f.write("\n")
        states_f.close()


# config1 = configuration()
# time step default = 0.01, number of steps default = 100
# dimensions default = 2, number of sample default = 50

# config1.setSteps(100)
# config1.setSamples(10)

# config1.setDynamics('Vanderpol')
# config1.setLowerBound([1.0, 1.0])
# config1.setUpperBound([50.0, 50.0])

# config1.setDynamics('Brussellator')
# config1.setLowerBound([1.0, 1.0])
# config1.setUpperBound([2.0, 2.0])

# config1.setDynamics('Lorentz')
# config1.setLowerBound([1.0, 1.0, 1.0])
# config1.setUpperBound([10.0, 10.0, 10.0])

# for i in range(0, 1):

# config1.storeTrajectories()
# config1.rawAnalyze()
# print (config1.eigenElbow)
# print (config1.noProminetEigVals)

# for j in range(0, config1.noProminetEigVals):
# print (config1.sortedVectors[j])
# config1.showLogEigenValues()
# config1.showTrajs()
