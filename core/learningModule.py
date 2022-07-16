import sys
from NNConfiguration import NNConfiguration
from itertools import combinations
from configuration import configuration
import os.path
from os import path
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, Activation, LeakyReLU
from tensorflow.keras.losses import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, hinge
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
# import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from frechet import norm
import matplotlib.pyplot as plt
from tensorflow.keras.layers import BatchNormalization
import random
import copy
import re
from tensorflow.keras.layers.experimental import RandomFourierFeatures
# Custom activation function
# from tensorflow.keras.utils.generic_utils import get_custom_objects
from rbflayer import RBFLayer, InitCentersRandom
from kmeans_initializer import InitCentersKMeans
nxg_path = os.environ.get('NXG_PATH')
sys.path.append(nxg_path + 'configurtion-setup/')

avg_v_val = 1.0


class DataConfiguration(configuration):

    def __init__(self, stepSize=0.01, dynamics='None', gradient_run=False, dimensions=2, sensitivity='Inv', disc=False,
                 normalization=False, validation=False):

        configuration.__init__(self, stepSize=stepSize, dynamics=dynamics, dimensions=dimensions,
                               gradient_run=gradient_run, disc=disc)
        self.data = []
        self.dimensions = dimensions
        self.eval_dir = nxg_path + 'eval-emsoft'
        self.sensitivity = sensitivity
        self.normalization = normalization
        self.validation = validation

    def getNormalizationStatus(self):
        return self.normalization

    def setNormalizationStatus(self, norm_status):
        self.normalization = norm_status

    def dumpDataConfiguration(self):
        d_obj_f_name = self.eval_dir + '/dconfigs_inv/d_object_' + self.dynamics
        print(d_obj_f_name)
        if self.sensitivity == 'Fwd':
            d_obj_f_name = self.eval_dir + '/dconfigs_fwd/d_object_' + self.dynamics

        d_obj_f_name += '.txt'
        if path.exists(d_obj_f_name):
            os.remove(d_obj_f_name)
        d_obj_f = open(d_obj_f_name, 'w')
        if self.discrete_dyn is True:
            d_obj_f.write(str(1) + '\n')
        else:
            d_obj_f.write(str(0) + '\n')
        if self.grad_run is True:
            d_obj_f.write(str(1) + '\n')
        else:
            d_obj_f.write(str(0) + '\n')
        d_obj_f.write(str(self.dimensions)+'\n')
        d_obj_f.write(str(self.steps)+'\n')
        d_obj_f.write(str(self.samples)+'\n')
        d_obj_f.write(str(self.stepSize)+'\n')
        for val in self.lowerBoundArray:
            d_obj_f.write(str(val))
            d_obj_f.write('\n')
        for val in self.upperBoundArray:
            d_obj_f.write(str(val))
            d_obj_f.write('\n')
        d_obj_f.close()

    def createData4GradRun(self, jumps):
        n_neighbors = int((len(self.trajectories) - self.samples)/self.samples)
        print(n_neighbors)
        idx = 0
        steps = len(self.trajectories[0]) - 1
        xList = []
        xpList = []
        vList = []
        vpList = []
        tList = []
        vp_norms = []
        while idx < len(self.trajectories):
            ref_traj = self.trajectories[idx]
            idy = 0
            for idy in range(1, n_neighbors+1):
                neighbor_traj = self.trajectories[idx+idy]
                x_idx = 0
                v_val = neighbor_traj[x_idx] - ref_traj[x_idx]
                v_norm = norm(v_val, 2)
                # We normalize them later during trainTestNN method since we save actual_inv_sen values for validation
                if self.validation is False:
                    print( " v _ val ")
                    v_val = [val / v_norm for val in v_val]
                x_val = ref_traj[x_idx]
                x_dv_val = neighbor_traj[x_idx]
                v_dv_val = ref_traj[x_idx] - neighbor_traj[x_idx]
                # v_dv_val = [val / v_norm for val in v_dv_val]

                for jump in jumps:
                    for step in range(1, (steps - jump), jump):
                        t_val = step
                        vp_val = neighbor_traj[t_val] - ref_traj[t_val]
                        vp_norm = norm(vp_val, 2)
                        vp_norms.append(vp_norm)
                        vp_val = [val / v_norm for val in vp_val]
                        xpList.append(ref_traj[t_val])
                        xList.append(x_val)
                        vList.append(v_val)
                        vpList.append(vp_val)
                        # print(vp_val)
                        tList.append(t_val * self.stepSize)

                        xp_dv_val = neighbor_traj[t_val]
                        vp_dv_val = ref_traj[t_val] - neighbor_traj[t_val]
                        vp_dv_val = [val / v_norm for val in vp_dv_val]
                        xpList.append(xp_dv_val)
                        xList.append(x_dv_val)
                        vpList.append(vp_dv_val)
                        vList.append(v_dv_val)
                        tList.append(t_val * self.stepSize)

            idx = idx + idy + 1

        xList = np.asarray(xList)
        xpList = np.asarray(xpList)
        vList = np.asarray(vList)
        vpList = np.asarray(vpList)
        tList = np.asarray(tList)

        print("Avg vp is " + str(sum(vp_norms)/len(vp_norms)))
        print("Max vp is " + str(max(vp_norms)))

        self.data.append(xList.tolist())
        self.data.append(xpList.tolist())
        self.data.append(vList.tolist())
        self.data.append(vpList.tolist())
        self.data.append(tList.tolist())

    def createData(self, dim=-1, jumps=[1], validation=False):
        assert self.lowerBoundArray is not [] and self.upperBoundArray is not []
        assert dim < self.dimensions

        if self.grad_run is True:
            self.normalization = True
            if validation is False:
                self.eval_dir = self.eval_dir + '/eval-gr'
                self.dumpDataConfiguration()
            return self.createData4GradRun(jumps)

        self.eval_dir = self.eval_dir + '/eval-non-gr'

        if validation is False:
            self.dumpDataConfiguration()
        traj_combs = []

        end_idx = self.samples
        start_idx = 0
        traj_indices = list(range(start_idx, end_idx))
        traj_combs += list(combinations(traj_indices, 2))
        print(traj_indices)
        steps = len(self.trajectories[traj_combs[0][0]]) - 1
        xList = []
        xpList = []
        vList = []
        vpList = []
        tList = []
        vp_norms = []
        if self.normalization is True:
            print("normalized")
        for traj_pair in traj_combs:
            t_pair = list(traj_pair)
            traj_1 = self.trajectories[t_pair[0]]
            traj_2 = self.trajectories[t_pair[1]]
            x_idx = 0
            v_val = traj_2[x_idx] - traj_1[x_idx]
            # v_norm = norm(v_val, 2)
            x_val = traj_1[x_idx]
            x_dv_val = traj_2[x_idx]
            v_dv_val = traj_1[x_idx] - traj_2[x_idx]
            for jump in jumps:
                for step in range(1, (steps - jump), jump):
                    t_val = step
                    xList.append(x_val)
                    vList.append(v_val)
                    xpList.append(traj_1[t_val])
                    vp_val = traj_2[t_val] - traj_1[t_val]
                    vp_norm = norm(vp_val, 2)
                    vp_norms.append(vp_norm)
                    vpList.append(vp_val)
                    tList.append(t_val * self.stepSize)

                    xList.append(x_dv_val)
                    vList.append(v_dv_val)
                    xpList.append(traj_2[t_val])
                    vp_dv_val = traj_1[t_val] - traj_2[t_val]
                    vpList.append(vp_dv_val)
                    tList.append(t_val * self.stepSize)

        xList = np.asarray(xList)
        xpList = np.asarray(xpList)
        vList = np.asarray(vList)
        vpList = np.asarray(vpList)
        tList = np.asarray(tList)

        print("Avg vp is " + str(sum(vp_norms)/len(vp_norms)))
        print("Max vp is " + str(max(vp_norms)))

        self.data.append(xList.tolist())
        self.data.append(xpList.tolist())
        self.data.append(vList.tolist())
        self.data.append(vpList.tolist())
        self.data.append(tList.tolist())

    def getData(self):
        return self.data

    def getDimensions(self):
        return self.dimensions

    def getEvalDir(self):
        return self.eval_dir

    def getSensitivity(self):
        return self.sensitivity

    def getRandomDataPoints(self, num):
        data_points = []
        for val in range(num):
            idx = random.randint(0, len(self.data[0]) - 1)
            data_point = []
            data_point.append(self.data[0][idx])
            data_point.append(self.data[1][idx])
            data_point.append(self.data[2][idx])
            data_point.append(self.data[3][idx])
            data_point.append(self.data[4][idx])
            data_points.append(data_point)
        return data_points

    def createDataPoint(self, x_val, xp_val, v_val, vp_val, t_val):
        data_point = []
        data_point.append(x_val)
        data_point.append(xp_val)
        data_point.append(v_val)
        data_point.append(vp_val)
        data_point.append(t_val*self.stepSize)
        return data_point


class CreateTrainNN(NNConfiguration):

    def __init__(self, dynamics=None, dnn_rbf='dnn'):
        NNConfiguration.__init__(self, dnn_rbf=dnn_rbf)
        self.dimensions = None
        self.predict_var = None
        self.dynamics = dynamics
        self.eval_dir = None
        self.normalization = False
        self.data_object = None

    def createInputOutput(self, data_object, inp_vars, out_vars):
        # We use this for validation
        if self.data_object is None:
            self.data_object = data_object

        assert data_object.getDynamics() == self.dynamics

        if data_object.getNormalizationStatus():
            self.normalization = True

        sensitivity = data_object.getSensitivity()
        for var in out_vars:
            if var == 'v' and sensitivity == 'fwd':
                print("Can not output v for forward sensitivity run. Either change sensitivity type or output var.")
                return
            elif var == 'vp' and sensitivity == 'inv':
                print("Can not output vp for inverse sensitivity run. Either change sensitivity type or output var.")
                return

        self.eval_dir = data_object.getEvalDir()
        print(self.eval_dir)
        print("Gradient: " + str(data_object.getGradientRun()))
        print("Normalization: " + str(data_object.getNormalizationStatus()))
        inp_indices = []
        out_indices = []
        data = data_object.getData()
        self.dimensions = data_object.getDimensions()

        for var in inp_vars:
            if var == 'x':
                inp_indices.append(0)
            if var == 'xp':
                inp_indices.append(1)
            if var == 'v':
                inp_indices.append(2)
            if var == 'vp':
                inp_indices.append(3)
            if var == 't':
                inp_indices.append(4)

        for var in out_vars:
            if var == 'v':
                out_indices.append(2)
                self.predict_var = var
            elif var == 'vp':
                out_indices.append(3)
                self.predict_var = var

        self.setInputSize((len(inp_indices)-1) * self.dimensions + 1)
        self.setOutputSize(self.dimensions)

        input = []
        output = []
        dataCount = len(data[0])
        for idx in range(dataCount):
            input_pair = []
            output_pair = []
            for inp in inp_indices:
                if inp != 4:
                    input_pair = input_pair + list(data[inp][idx])
                else:
                    input_pair = input_pair + [data[inp][idx]]
            for out in out_indices:
                output_pair = data[out][idx]
            input.append(input_pair)
            output.append(output_pair)
        # print(input.shape)
        self.setInput(np.asarray(input, dtype=np.float64))
        self.setOutput(np.asarray(output, dtype=np.float64))

    def trainTestNN(self, optim='SGD', loss_fn='mae', act_fn='ReLU', layers=4, neurons=400, validation=False):

        print(self.input.shape)
        x_train, x_test, y_train, y_test = train_test_split(self.input, self.output, test_size=self.test_size,
                                                            shuffle=True, random_state=1)
        print(x_train.shape)
        print(y_train.shape)
        # y_train = y_train.tolist()
        # print(y_train[0].shape)

        actual_inv_sens = []

        if self.normalization is True and validation is True:
            print(" validation and normalized ")
            for idx in range(len(y_train)):
                y_train[idx] = [val / norm(y_train[idx], 2) for val in y_train[idx]]

            for idx in range(len(y_test)):
                actual_inv_sens.append(y_test[idx].copy())

            for idx in range(len(y_test)):
                y_test[idx] = [val / norm(y_test[idx], 2) for val in y_test[idx]]

        # for idx in range(len(actual_inv_sens)):
        #     print(norm(actual_inv_sens[idx], 2))

        inputs_train = x_train
        targets_train = y_train
        inputs_test = x_test
        targets_test = y_test

        def swish(x):
            return K.sigmoid(x) * 5

        #   get_custom_objects().update({'custom_activation': Activation(swish)})

        if act_fn == 'Tanh':
            act = 'tanh'
        elif act_fn == 'Sigmoid':
            act = 'sigmoid'
        elif act_fn == 'Exponential':
            act = 'exponential'
        elif act_fn == 'Linear':
            act = 'linear'
        elif act_fn == 'SoftMax':
            act = 'softmax'
        elif act_fn == 'Swish':
            act = swish
        else:
            act = 'relu'
            print("\nSetting the activation function to default - ReLU.\n")

        def mre_loss(y_true, y_pred):
            # loss = K.mean(K.square(y_pred - y_true)/K.square(y_pred))
            # loss = K.mean(K.abs((y_true - y_pred) / K.clip(K.abs(y_pred), K.epsilon(), np.inf)))
            # my_const = tf.constant(0.0001, dtype=tf.float32)
            # loss = K.mean(K.square(y_pred - y_true) / (K.square(y_pred)+3))
            # loss = K.mean(K.square(y_pred - y_true) / (K.square(y_pred) + 2))
            # loss = tf.reduce_mean(tf.divide((tf.subtract(y_pred, y_true))**2, (y_pred**2 + 1e-10)))

            loss_1 = K.sum(K.mean(K.square(y_pred - y_true) / (K.square(y_pred) + 4), axis=1))
            loss_0 = K.sum(K.mean(K.square(y_pred - y_true) / (K.square(y_pred) + 4), axis=0))

            bool_idx = K.greater((loss_1 - loss_0), 0)

            # Vanderpol, Jetengine. Buckling
            loss = K.switch(bool_idx, loss_1 * 1.4 + loss_0 * 0.9, loss_1 * 0.9 + loss_0 * 1.4)

            # Spring Pendulum and rest, if not specified
            # loss = K.switch(bool_idx, loss_1 * 2.0 + loss_0 * 1.3, loss_1 * 1.3 + loss_0 * 2.0)

            return loss

        model = Sequential()

        # Dense(64) is a fully-connected layer with 64 hidden units.
        # in the first layer, you must specify the expected input data shape:
        # here, 20-dimensional vectors.

        if optim == 'Adam':
            optimizer = Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
        elif optim == 'RMSProp':
            optimizer = RMSprop()
        else:
            optimizer = SGD(learning_rate=self.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)

        print("***** learning module ***" + str(self.input_size) + str(self.output_size))

        if self.DNN_or_RBF == 'dnn':
            model.add(Dense(neurons, activation=act, input_dim=self.input_size))
            start_neurons = neurons
            for idx in range(layers):
                model.add(Dense(start_neurons, activation=act))
                # start_neurons = start_neurons/2
                model.add(BatchNormalization())
            model.add(Dense(self.output_size, activation='linear'))

            # if optim is 'Adam':
            #     optimizer = Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
            # elif optim is 'RMSProp':
            #     optimizer = RMSprop(learning_rate=self.learning_rate, rho=0.9)
            # else:
            #     optimizer = SGD(learning_rate=self.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)

            if loss_fn == 'mse':
                model.compile(loss=mean_squared_error, optimizer=optimizer, metrics=['accuracy', 'mae'])
            elif loss_fn == 'mae':
                model.compile(loss=mean_absolute_error, optimizer=optimizer, metrics=['accuracy', 'mse'])
            elif loss_fn == 'mape':
                model.compile(loss=mean_absolute_percentage_error, optimizer=optimizer, metrics=['accuracy', 'mse'])
            elif loss_fn == 'mre':
                model.compile(loss=mre_loss, optimizer=optimizer, metrics=['accuracy', 'mse'])

            model.fit(inputs_train, targets_train,
                      epochs=self.epochs,
                      batch_size=self.batch_size, verbose=1)
        # score = model.evaluate(self.x_test, self.y_test, batch_size=self.batch_size)
        # print(score)

        elif self.DNN_or_RBF == 'fourier':
            model = Sequential(
                [
                    Input(shape=(self.input_size,)),
                    RandomFourierFeatures(
                        output_dim=self.output_size, scale=10.0, kernel_initializer="gaussian"
                    ),
                    Dense(units=self.output_size, activation='linear'),
                ]
            )
            model.compile(
                optimizer=Adam(learning_rate=1e-3),
                loss=hinge,
                metrics=[mean_squared_error],
            )

            model.fit(inputs_train, targets_train,
                      epochs=self.epochs,
                      batch_size=self.batch_size, verbose=1)

        elif self.DNN_or_RBF == 'RBF':

            # https://github.com/PetraVidnerova/rbf_for_tf2
            model = Sequential()
            rbflayer = RBFLayer(neurons,
                                initializer=InitCentersRandom(inputs_train),
                                betas=0.9,  # Lower the better. Tried with 10 and 0.5, but it wasn't good
                                input_shape=(self.input_size,))
            outputlayer = Dense(self.output_size, activation='linear', use_bias=True)

            model.add(rbflayer)
            # model.add(Dense(neurons))
            # model.add(LeakyReLU(alpha=0.1))
            start_neurons = neurons
            for idx in range(layers):
                model.add(Dense(start_neurons, activation=act))
                # start_neurons = start_neurons/2
                model.add(BatchNormalization())
            model.add(outputlayer)

            if loss_fn == 'mse':
                model.compile(loss=mean_squared_error, optimizer=optimizer, metrics=['accuracy', 'mae'])
            elif loss_fn == 'mae':
                model.compile(loss=mean_absolute_error, optimizer=optimizer, metrics=['accuracy', 'mse'])
            elif loss_fn == 'mape':
                model.compile(loss=mean_absolute_percentage_error, optimizer=optimizer, metrics=['accuracy', 'mse'])
            elif loss_fn == 'mre':
                model.compile(loss=mre_loss, optimizer=optimizer, metrics=['accuracy', 'mse'])

            # model.compile(loss='mean_absolute_error',
            #               optimizer=RMSprop(), metrics=['accuracy', 'mse'])  # mae is better

            # print(self.batch_size)
            # fit and predict
            model.fit(inputs_train, targets_train,
                      batch_size=self.batch_size,
                      epochs=self.epochs,
                      verbose=1)

        # print(model.summary())
        # for idx in range(len(model.layers)):
        #     print(model.layers[idx].get_config())
        predicted_train = model.predict(inputs_train)
        print(predicted_train.shape)

        if self.predict_var == 'v':
            v_f_name = self.eval_dir + "/models/model_vp_2_v_"
            v_f_name = v_f_name + str(self.dynamics)
            v_f_name = v_f_name + "_" + self.DNN_or_RBF
            # weights_f_name = v_f_name
            # weights_f_name = weights_f_name + ".yaml"
            v_f_name = v_f_name + "_" + str(layers)
            v_f_name = v_f_name + "_" + str(neurons)
            v_f_name = v_f_name + "_" + str(act_fn)
            # if self.normalization is True:
            #     v_f_name = v_f_name + "_norm"
            v_f_name = v_f_name + ".h5"

            if validation is False:
                if path.exists(v_f_name):
                    os.remove(v_f_name)
                model.save(v_f_name)
            # model_yaml = model.to_yaml()
            # with open(weights_f_name, "w") as yaml_file:
            #     yaml_file.write(model_yaml)
        elif self.predict_var == 'vp':
            vp_f_name = self.eval_dir + "/models/model_v_2_vp_"
            vp_f_name = vp_f_name + str(self.dynamics)
            vp_f_name = vp_f_name + "_" + self.DNN_or_RBF
            vp_f_name = vp_f_name + "_" + str(layers)
            vp_f_name = vp_f_name + "_" + str(neurons)
            vp_f_name = vp_f_name + "_" + str(act_fn)
            # if self.normalization is True:
            #     vp_f_name = vp_f_name + "_norm"
            vp_f_name = vp_f_name + ".h5"

            if validation is False:
                if path.exists(vp_f_name):
                    os.remove(vp_f_name)
                model.save(vp_f_name)
            model.save(vp_f_name)

        predicted_test = model.predict(inputs_test)

        max_se_train = 0.0
        min_se_train = 1000.0
        mse_train = 0.0
        for idx in range(len(targets_train)):
            dist_val = norm(predicted_train[idx] - targets_train[idx], 2)
            if dist_val > max_se_train:
                max_se_train = dist_val
            if dist_val < min_se_train:
                min_se_train = dist_val
            mse_train += dist_val
        mse_train = mse_train / (len(targets_train))

        min_se_test = 1000.0
        max_se_test = 0.0
        mse_test = 0.0
        for idx in range(len(targets_test)):
            dist_val = norm(predicted_test[idx] - targets_test[idx], 2)
            if dist_val > max_se_test:
                max_se_test = dist_val
            if dist_val < min_se_test:
                min_se_test = dist_val
            mse_test += dist_val
        mse_test = mse_test / (len(targets_test))

        # print("Max RMSE Train {}".format(max_se_train))
        # print("Max RMSE Test {}".format(max_se_test))
        # print("Min RMSE Train {}".format(min_se_train))
        # print("Min RMSE Test {}".format(min_se_test))

        # print("Mean RMSE Train {}".format(mse_train))
        # print("Mean RMSE Test {}".format(mse_test))

        max_re_train = 0.0
        mre_train = 0.0
        for idx in range(len(targets_train)):
            dist_val = norm(predicted_train[idx] - targets_train[idx], 2)
            dist_val = (dist_val / (norm(targets_train[idx], 2)))
            if dist_val > max_re_train:
                max_re_train = dist_val
            mre_train += dist_val
        mre_train = mre_train / (len(targets_train))

        max_re_test = 0.0
        mre_test = 0.0
        for idx in range(len(targets_test)):
            dist_val = norm(predicted_test[idx] - targets_test[idx], 2)
            dist_val = (dist_val / (norm(targets_test[idx], 2)))
            if dist_val > max_re_test:
                max_re_test = dist_val
            mre_test += dist_val
        mre_test = mre_test / (len(targets_test))
        # print("Max Relative Error Train {}".format(max_re_train))
        # print("Max Relative Error Test {}".format(max_re_test))

        print("Mean Relative Error Train {}".format(mre_train))
        print("Mean Relative Error Test {}".format(mre_test))

        self.visualizePerturbation(targets_train, predicted_train)
        self.visualizePerturbation(targets_test, predicted_test)

        if validation is True:
            # self.visualizeDelta(targets_test, predicted_test, actual_inv_sens)
            for delta_or_mse in ['delta']:
                # self.validateNetwork(model, 0.9, delta_or_mse)
                # self.validateNetwork(model, 0.8, delta_or_mse)
                # self.validateNetwork(model, 0.7, delta_or_mse)
                self.validateNetwork(model, 0.5, delta_or_mse)
                # self.validateNetwork(model, 0.4, delta_or_mse)
                self.validateNetwork(model, 0.3, delta_or_mse)
                # self.validateNetwork(model, 0.2, delta_or_mse)
                self.validateNetwork(model, 0.1, delta_or_mse)
                self.validateNetwork(model, 0.09, delta_or_mse)
                self.validateNetwork(model, 0.08, delta_or_mse)
                # self.validateNetwork(model, 0.07, delta_or_mse)
                self.validateNetwork(model, 0.06, delta_or_mse)
                # self.validateNetwork(model, 0.05, delta_or_mse)
                self.validateNetwork(model, 0.04, delta_or_mse)
                # self.validateNetwork(model, 0.03, delta_or_mse)
                self.validateNetwork(model, 0.02, delta_or_mse)
                self.validateNetwork(model, 0.01, delta_or_mse)
                self.validateNetwork(model, 0.009, delta_or_mse)
                self.validateNetwork(model, 0.007, delta_or_mse)
                self.validateNetwork(model, 0.005, delta_or_mse)
                self.validateNetwork(model, 0.003, delta_or_mse)
                self.validateNetwork(model, 0.001, delta_or_mse)
        # self.visualizeEpsilon(targets_test, predicted_test)

    def visualizePerturbation(self, t, p):
        # targets = t.detach().numpy()
        # predicted = tf.Session().run(p)
        targets = t
        predicted = p
        print(targets.shape)
        print(predicted.shape)
        t_shape = targets.shape
        for dim in range(self.dimensions):

            y_test_plt = []
            predicted_test_plt = []

            if t_shape[0] < 2000:
                print_range = t_shape[0]-1
            else:
                print_range = 2000
            for idx in range(0, print_range):
                y_test_plt += [targets[idx][dim]]
                predicted_test_plt += [predicted[idx][dim]]

            plt.figure()
            plt.plot(y_test_plt)
            plt.plot(predicted_test_plt)
            plt.show()

        # if self.predict_var is 'v':
        #     f_name = self.eval_dir + 'outputs/v_vals_'
        # else:
        #     f_name = self.eval_dir + 'outputs/vp_vals_'
        # f_name = f_name + self.dynamics
        # f_name = f_name + "_" + self.DNN_or_RBF
        # if self.dynamics == "AeroBench":
        #     f_name = f_name + "_"
        #     f_name = f_name + str(self.dimensions)
        # f_name = f_name + ".txt"
        # if path.exists(f_name):
        #     os.remove(f_name)
        # vals_f = open(f_name, "w")

        # for idx in range(0, t_shape[0]-1):
        #     vals_f.write(str(targets[idx]))
        #     vals_f.write(" , ")
        #     vals_f.write(str(predicted[idx]))
        #     vals_f.write(" ... ")
        #     t_norm = norm(targets[idx], 2)
        #     vals_f.write(str(t_norm))
        #     vals_f.write(" , ")
        #     p_norm = norm(predicted[idx], 2)
        #     vals_f.write(str(p_norm))
        #     vals_f.write("\n")
        #
        # vals_f.close()

    def visualizeDelta(self, target, prediction, actual_vals):
        # global avg_v_val
        # print(avg_v_val)
        # epsilons = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05,
        # 0.06, 0.07, 0.08, 0.09, 0.1, 0.2]
        epsilon = 0.0001
        deltas = []

        # target is the unit direction (after normalization) - needed for training
        # actual_vals is the original inverse sensitivity before normalization
        if self.normalization is True:
            print(" visualizeDelta - normalize ")
            assert len(target) == len(actual_vals)
            for idx in range(len(target)):
                v_norm = norm(actual_vals[idx], 2)
                dist_val = norm(prediction[idx] * v_norm - actual_vals[idx], 2)
                delta = (dist_val - epsilon * v_norm)
                deltas.append(delta)
        else:
            for idx in range(len(target)):
                # dist_val = norm(prediction[idx] - target[idx], 2)
                # delta = (dist_val - epsilon * norm(target[idx], 2))
                dist_val = norm(prediction[idx] - target[idx], 2)
                delta = (dist_val - epsilon * norm(target[idx], 2))
                deltas.append(delta)
        deltas.sort()
        delta_sum = sum(deltas)
        quantile_10_idx = int(0.10 * len(deltas))
        quantile_25_idx = int(0.25 * len(deltas))
        quantile_75_idx = int(0.75 * len(deltas))
        quantile_90_idx = int(0.9 * len(deltas))
        print(" ************ deltas ************")
        print(" *** First val " + str(deltas[0]))
        print(" *** 10 quantile " + str(deltas[quantile_10_idx]))
        print(" *** 25 quantile " + str(deltas[quantile_25_idx]))
        print(" *** mean " + str(delta_sum/len(deltas)))
        print(" *** 75 quantile " + str(deltas[quantile_75_idx]))
        print(" *** 90 quantile " + str(deltas[quantile_90_idx]))
        print(" *** Last val " + str(deltas[len(deltas)-1]))
        # print(deltas)

    def visualizeMSE(self, target, prediction, actual_vals):
        # global avg_v_val
        # print(avg_v_val)
        mse_vals = []
        # target is the unit direction (after normalization) - needed for training
        # actual_vals is the original inverse sensitivity before normalization
        if self.normalization is True:
            print("here")
            assert len(target) == len(actual_vals)
            for idx in range(len(target)):
                v_norm = norm(actual_vals[idx], 2)
                actual_vals_dir = (actual_vals[idx]/v_norm)
                mse_val = norm(prediction[idx] - actual_vals_dir, 2)
                mse_vals.append(mse_val)
        else:
            for idx in range(len(target)):
                # dist_val = norm(prediction[idx] - target[idx], 2)
                # delta = (dist_val - epsilon * norm(target[idx], 2))
                mse_val = norm(prediction[idx] - target[idx], 2)
                mse_vals.append(mse_val)
        mse_vals.sort()
        mse_sum = sum(mse_vals)
        quantile_10_idx = int(0.10 * len(mse_vals))
        quantile_25_idx = int(0.25 * len(mse_vals))
        quantile_75_idx = int(0.75 * len(mse_vals))
        quantile_90_idx = int(0.9 * len(mse_vals))
        print(" ************ MSE ************")
        print(" *** First val " + str(mse_vals[0]))
        print(" *** 10 quantile " + str(mse_vals[quantile_10_idx]))
        print(" *** 25 quantile " + str(mse_vals[quantile_25_idx]))
        print(" *** mean " + str(mse_sum/len(mse_vals)))
        print(" *** 75 quantile " + str(mse_vals[quantile_75_idx]))
        print(" *** 90 quantile " + str(mse_vals[quantile_90_idx]))
        print(" *** Last val " + str(mse_vals[len(mse_vals)-1]))
        # print(deltas)

    def visualizeEpsilon(self, target, prediction):
        # global avg_v_val
        # print(avg_v_val)
        deltas = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        epsilons = []
        for delta in deltas:
            epsilon = 0.0
            for idx in range(len(target)):
                # dist_val = norm(prediction[idx] - target[idx], 2)
                # epsilon += (dist_val * 0.001 - delta)/(norm(target[idx], 2))
                dist_val = norm(prediction[idx] - target[idx], 2)
                epsilon += (dist_val - delta * norm(target[idx], 2))
            epsilons.append(epsilon / len(target))
        print(epsilons)
        plt.figure()
        plt.plot(deltas, epsilons)
        plt.show()

    def validateNetwork(self, network, scale, delta_or_mse):

        # print("Length of local data object trajectories " + str(len(self.data_object.trajectories)))
        print(" ******* scale ******* " + str(scale))
        self.data_object.trajectories = []
        r_states = self.data_object.load_i_states_4m_file()
        self.data_object.setSamples(len(r_states))
        self.data_object.grad_run = True
        self.data_object.generateTrajectories(scaling=scale, r_states=r_states)
        print("Length of local data object trajectories " + str(len(self.data_object.trajectories)))

        self.data_object.data = []
        self.data_object.createData(jumps=[1, 2, 5, 7, 11, 13, 17, 19], validation=True)

        # self.input.clear()
        # self.output.clear()
        self.createInputOutput(data_object=self.data_object, inp_vars=['x','xp','vp','t'], out_vars=['v'])
        x_train, x_test, y_train, y_test = train_test_split(self.input, self.output, test_size=0.95, shuffle=True,
                                                            random_state=1)
        actual_inv_sens = []
        print(norm(y_test[0], 2))

        if self.normalization is True:
            print("normalized")
            for idx in range(len(y_train)):
                y_train[idx] = [val / norm(y_train[idx], 2) for val in y_train[idx]]

            for idx in range(len(y_test)):
                actual_inv_sens.append(y_test[idx].copy())

            for idx in range(len(y_test)):
                y_test[idx] = [val / norm(y_test[idx], 2) for val in y_test[idx]]

        inputs_test = x_test
        targets_test = y_test

        predicted_test = network.predict(inputs_test)
        if delta_or_mse == 'delta':
            self.visualizeDelta(targets_test, predicted_test, actual_inv_sens)
        elif delta_or_mse == 'mse':
            self.visualizeMSE(targets_test, predicted_test, actual_inv_sens)
        elif delta_or_mse == ['delta', 'mse']:
            self.visualizeDelta(targets_test, predicted_test, actual_inv_sens)
            self.visualizeMSE(targets_test, predicted_test, actual_inv_sens)



